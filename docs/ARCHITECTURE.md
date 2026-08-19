# Five-layer CR-IoT implementation map

1. **Perception:** IoT sensors and RF front ends collect environmental and complex-IQ samples.
2. **Edge sensing:** feature extraction and fast spectrum-occupancy decisions run locally.
3. **Network:** MQTT/CoAP/REST transports occupancy and quality metadata, not necessarily raw IQ.
4. **Service:** fleet aggregation, model training, monitoring, and spectrum-policy services run in the cloud.
5. **Application:** channel-selection, coexistence, dashboards, and alerts consume validated decisions.

For safety, the included package stops at local classification and offline benchmarking. It does not transmit on licensed spectrum or control radios.

