# Inference output message

std_msgs/Header header

# Flat action/prediction vector
std_msgs/Float32MultiArray data

# Name of the output (matches model.output_names)
string output_name

# Mirrors the seq_id from the request for latency measurement
uint64 seq_id

# Inference latency in milliseconds (set by the node)
float32 latency_ms
