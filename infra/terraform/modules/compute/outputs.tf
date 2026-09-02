output "instance_id" {
  description = "EC2 instance ID (use it as the SSM Session Manager target)."
  value       = aws_instance.this.id
}

output "public_ip" {
  description = "Elastic IP attached to the instance."
  value       = aws_eip.this.public_ip
}

output "security_group_id" {
  description = "ID of the instance security group."
  value       = aws_security_group.this.id
}
