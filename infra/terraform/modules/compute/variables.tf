variable "name_prefix" {
  description = "Prefix applied to the Name tag of every compute resource."
  type        = string
}

variable "vpc_id" {
  description = "VPC the security group is created in."
  type        = string
}

variable "subnet_id" {
  description = "Public subnet the instance is launched into."
  type        = string
}

variable "ami_id" {
  description = "AMI ID for the instance (Amazon Linux 2023 x86_64)."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type. Ollama needs real memory; t3.large (8 GB) is the practical floor."
  type        = string
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB (models + images need headroom)."
  type        = number
}

variable "user_data" {
  description = "Rendered cloud-init script that installs Docker and starts the stack."
  type        = string
}

variable "api_ingress_cidr" {
  description = "CIDR allowed to reach the RAG API on port 8000."
  type        = string
}

variable "grafana_ingress_cidr" {
  description = "CIDR allowed to reach Grafana on port 3000. Empty string keeps it closed."
  type        = string
  default     = ""
}

variable "enable_ssh" {
  description = "Open inbound SSH (22). Off by default; prefer SSM Session Manager."
  type        = bool
  default     = false
}

variable "ssh_admin_cidr" {
  description = "CIDR allowed to SSH when enable_ssh is true. Never use 0.0.0.0/0."
  type        = string
  default     = "127.0.0.1/32"
}

variable "ssh_key_name" {
  description = "Existing EC2 key pair name for SSH. Empty string launches with no key pair."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags merged into every resource."
  type        = map(string)
  default     = {}
}
