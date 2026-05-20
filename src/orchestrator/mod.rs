// Orchestrator (sequential, parallel, routing)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tracing::Instrument;

use crate::agent::Agent;
use crate::error::KovaError;

/// Describes how agents should be executed in an orchestration workflow.
pub enum OrchestratorPattern {
    /// Execute agents in order, chaining output of agent i as input to agent i+1.
    Sequential(Vec<String>),
    /// Execute all agents concurrently with the same input.
    Parallel(Vec<String>),
    /// Send input to a router agent, parse its response to select a downstream agent.
    Router {
        router_agent: String,
        downstream: Vec<String>,
    },
}

/// Result of a parallel orchestration, containing successes and failures.
pub struct ParallelResult {
    pub successes: Vec<(String, String)>,
    pub failures: Vec<(String, KovaError)>,
}

/// Coordinates multiple [`Agent`] instances in sequential, parallel, or
/// routing workflows with a configurable timeout.
pub struct Orchestrator {
    agents: HashMap<String, Arc<Agent>>,
    timeout: Duration,
}

impl Orchestrator {
    /// Create a new orchestrator with the given agents and workflow timeout.
    pub fn new(agents: HashMap<String, Arc<Agent>>, timeout: Duration) -> Self {
        Self { agents, timeout }
    }

    /// Execute the given orchestration pattern with the provided input.
    pub async fn execute(
        &self,
        pattern: OrchestratorPattern,
        input: &str,
    ) -> Result<OrchestratorOutput, KovaError> {
        let pattern_type = match &pattern {
            OrchestratorPattern::Sequential(_) => "sequential",
            OrchestratorPattern::Parallel(_) => "parallel",
            OrchestratorPattern::Router { .. } => "router",
        };
        let span = tracing::info_span!(
            "orchestrator.execute",
            pattern = pattern_type,
            otel.status_code = tracing::field::Empty,
        );

        async {
            let result = tokio::time::timeout(self.timeout, self.execute_inner(pattern, input))
                .await
                .map_err(|_| {
                    tracing::Span::current().record("otel.status_code", "ERROR");
                    tracing::warn!("Orchestration timed out");
                    KovaError::Orchestration(format!(
                        "Orchestration timed out after {:?}",
                        self.timeout
                    ))
                })?;
            if result.is_err() {
                tracing::Span::current().record("otel.status_code", "ERROR");
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn execute_inner(
        &self,
        pattern: OrchestratorPattern,
        input: &str,
    ) -> Result<OrchestratorOutput, KovaError> {
        match pattern {
            OrchestratorPattern::Sequential(agent_names) => {
                self.execute_sequential(&agent_names, input).await
            }
            OrchestratorPattern::Parallel(agent_names) => {
                self.execute_parallel(&agent_names, input).await
            }
            OrchestratorPattern::Router {
                router_agent,
                downstream,
            } => {
                self.execute_router(&router_agent, &downstream, input)
                    .await
            }
        }
    }

    fn get_agent(&self, name: &str) -> Result<&Arc<Agent>, KovaError> {
        self.agents.get(name).ok_or_else(|| {
            KovaError::Orchestration(format!("Agent '{}' not found in orchestrator", name))
        })
    }

    /// Sequential: chain output of agent i as input to agent i+1.
    async fn execute_sequential(
        &self,
        agent_names: &[String],
        input: &str,
    ) -> Result<OrchestratorOutput, KovaError> {
        if agent_names.is_empty() {
            return Err(KovaError::Orchestration(
                "Sequential pattern requires at least one agent".to_string(),
            ));
        }

        let mut current_input = input.to_string();
        for (i, name) in agent_names.iter().enumerate() {
            let agent = self.get_agent(name)?;
            let conversation_id = format!("orch-seq-{}-{}", name, i);
            current_input = agent
                .chat(&conversation_id, &current_input)
                .await
                .map_err(|e| {
                    KovaError::Orchestration(format!(
                        "Agent '{}' failed in sequential pipeline: {}",
                        name, e
                    ))
                })?;
        }

        Ok(OrchestratorOutput::Single(current_input))
    }

    /// Parallel: spawn all agents concurrently with the same input.
    /// Collects all results — successes and failures.
    async fn execute_parallel(
        &self,
        agent_names: &[String],
        input: &str,
    ) -> Result<OrchestratorOutput, KovaError> {
        if agent_names.is_empty() {
            return Err(KovaError::Orchestration(
                "Parallel pattern requires at least one agent".to_string(),
            ));
        }

        // Validate all agents exist before spawning.
        let agents: Vec<(String, Arc<Agent>)> = agent_names
            .iter()
            .map(|name| {
                self.get_agent(name)
                    .map(|a| (name.clone(), Arc::clone(a)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut handles = Vec::with_capacity(agents.len());
        for (name, agent) in agents {
            let input_owned = input.to_string();
            let conv_id = format!("orch-par-{}", name);
            let agent_span = tracing::info_span!(
                "orchestrator.agent",
                agent.name = %name,
            );
            handles.push(tokio::spawn(
                async move {
                    let result = agent.chat(&conv_id, &input_owned).await;
                    (name, result)
                }
                .instrument(agent_span),
            ));
        }

        let mut successes = Vec::new();
        let mut failures = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((name, Ok(output))) => successes.push((name, output)),
                Ok((name, Err(e))) => failures.push((name, e)),
                Err(join_err) => {
                    failures.push((
                        "unknown".to_string(),
                        KovaError::Orchestration(format!("Task join error: {}", join_err)),
                    ));
                }
            }
        }

        Ok(OrchestratorOutput::Parallel(ParallelResult {
            successes,
            failures,
        }))
    }

    /// Router: send input to router agent, parse response to select downstream agent.
    async fn execute_router(
        &self,
        router_name: &str,
        downstream: &[String],
        input: &str,
    ) -> Result<OrchestratorOutput, KovaError> {
        let router = self.get_agent(router_name)?;
        let conv_id = format!("orch-router-{}", router_name);

        // The router agent's response should be the name of a downstream agent.
        let selected_name = router
            .chat(&conv_id, input)
            .await
            .map_err(|e| {
                KovaError::Orchestration(format!("Router agent '{}' failed: {}", router_name, e))
            })?;

        let selected_name = selected_name.trim().to_string();

        // Validate the selected agent is in the downstream list.
        if !downstream.contains(&selected_name) {
            return Err(KovaError::Orchestration(format!(
                "Router selected '{}' which is not in downstream agents: {:?}",
                selected_name, downstream
            )));
        }

        let agent = self.get_agent(&selected_name)?;
        let downstream_conv_id = format!("orch-routed-{}", selected_name);
        let output = agent
            .chat(&downstream_conv_id, input)
            .await
            .map_err(|e| {
                KovaError::Orchestration(format!(
                    "Downstream agent '{}' failed: {}",
                    selected_name, e
                ))
            })?;

        Ok(OrchestratorOutput::Single(output))
    }
}

/// The output of an orchestration execution.
pub enum OrchestratorOutput {
    /// A single result from sequential or routing patterns.
    Single(String),
    /// Results from parallel execution, including any failures.
    Parallel(ParallelResult),
}
