output "instance_id" {
  description = "EC2 instance ID."
  value       = module.compute.instance_id
}

output "public_ip" {
  description = "Elastic IP of the instance."
  value       = module.compute.public_ip
}

output "api_url" {
  description = "Base URL of the RAG API once the stack is up."
  value       = "http://${module.compute.public_ip}:8000"
}

output "health_url" {
  description = "Liveness probe URL (no auth)."
  value       = "http://${module.compute.public_ip}:8000/health"
}

output "ssm_session_command" {
  description = "Open an audited shell on the instance without SSH."
  value       = "aws ssm start-session --target ${module.compute.instance_id}"
}

output "grafana_port_forward_command" {
  description = "Reach Grafana locally without exposing port 3000."
  value       = "aws ssm start-session --target ${module.compute.instance_id} --document-name AWS-StartPortForwardingSession --parameters '{\"portNumber\":[\"3000\"],\"localPortNumber\":[\"3000\"]}'"
}
