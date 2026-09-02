output "vpc_id" {
  description = "ID of the created VPC."
  value       = aws_vpc.this.id
}

output "public_subnet_id" {
  description = "ID of the public subnet hosting the instance."
  value       = aws_subnet.public.id
}

output "private_subnet_id" {
  description = "ID of the reserved private subnet."
  value       = aws_subnet.private.id
}
