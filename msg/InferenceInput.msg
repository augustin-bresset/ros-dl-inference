# Inference input message
# Carries a flat float array plus optional metadata

std_msgs/Header header

# Flat observation vector (reshape using layout)
std_msgs/Float32MultiArray data

# Optional: semantic name of the input (matches model.input_names)
string input_name

# Sequence number for latency tracking
uint64 seq_id
