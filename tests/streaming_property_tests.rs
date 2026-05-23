use proptest::prelude::*;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use kova::error::KovaError;
use kova::models::*;
use kova::streaming::StreamingHandler;

fn arb_stop_reason() -> impl Strategy<Value = StopReason> {
    prop_oneof![
        Just(StopReason::EndTurn),
        Just(StopReason::ToolUse),
        Just(StopReason::MaxTokens),
        "[a-z_]{3,15}".prop_map(StopReason::Unknown),
    ]
}

fn arb_stream_event() -> impl Strategy<Value = StreamEvent> {
    prop_oneof![
        "[a-zA-Z0-9 ]{0,50}".prop_map(|text| StreamEvent::ContentDelta { text }),
        (
            "[a-z0-9_]{1,20}",
            proptest::option::of("[a-z_]{1,20}"),
            proptest::option::of("[a-zA-Z0-9 {}]{0,30}"),
        )
            .prop_map(|(id, name, input_delta)| StreamEvent::ToolUseDelta {
                id,
                name,
                input_delta,
            }),
        arb_stop_reason().prop_map(|stop_reason| StreamEvent::StopEvent { stop_reason }),
        "[a-zA-Z0-9 ]{0,50}".prop_map(|message| StreamEvent::Error { message }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_streaming_chunk_order_preservation(
        events in proptest::collection::vec(arb_stream_event(), 1..20)
    ) {
        struct CollectingHandler {
            collected: Arc<Mutex<Vec<StreamEvent>>>,
        }

        #[async_trait]
        impl StreamingHandler for CollectingHandler {
            async fn on_chunk(&self, event: &StreamEvent) -> Result<(), KovaError> {
                self.collected.lock().await.push(event.clone());
                Ok(())
            }
            async fn on_complete(&self) -> Result<(), KovaError> {
                Ok(())
            }
            async fn on_error(&self, _error: &KovaError) {}
        }

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let collected = Arc::new(Mutex::new(Vec::new()));
            let handler = CollectingHandler {
                collected: collected.clone(),
            };

            for event in &events {
                handler.on_chunk(event).await.unwrap();
            }
            handler.on_complete().await.unwrap();

            let received = collected.lock().await;
            prop_assert_eq!(
                received.len(),
                events.len(),
                "expected {} events, got {}",
                events.len(),
                received.len()
            );

            for (i, (got, expected)) in received.iter().zip(events.iter()).enumerate() {
                prop_assert_eq!(
                    got, expected,
                    "event at index {} differs", i
                );
            }

            Ok(())
        })?;
    }
}
