# ── General ───────────────────────────────────────────────────────────────────
variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name, used for tags and resource name prefixes."
  type        = string
  default     = "aegis-rag"
}

variable "environment" {
  description = "Environment name (dev/staging/prod), used for tags and naming."
  type        = string
  default     = "dev"
}

# ── Network ───────────────────────────────────────────────────────────────────
variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet."
  type        = string
  default     = "10.20.1.0/24"
}

variable "private_subnet_cidr" {
  description = "CIDR block for the reserved private subnet."
  type        = string
  default     = "10.20.11.0/24"
}

# ── Compute ───────────────────────────────────────────────────────────────────
variable "instance_type" {
  description = <<-EOT
    EC2 instance type. Ollama (llama3.2) needs real memory, so t3.large (2 vCPU /
    8 GB) is the practical floor. This is NOT free-tier eligible — set it to
    t3.micro only for a no-cost smoke test that will not actually run the model.
  EOT
  type        = string
  default     = "t3.large"
}

variable "root_volume_gb" {
  description = "Root EBS (gp3) volume size in GB. Container images + Ollama models need headroom."
  type        = number
  default     = 40
}

# ── Access control ────────────────────────────────────────────────────────────
variable "api_ingress_cidr" {
  description = <<-EOT
    CIDR allowed to reach the RAG API on port 8000. The app enforces API-key auth
    and rate limiting, but restrict this to known callers as the first layer.
    Defaults open; narrow it before any real exposure.
  EOT
  type        = string
  default     = "0.0.0.0/0"
}

variable "grafana_ingress_cidr" {
  description = "CIDR allowed to reach Grafana (3000). Empty keeps it closed — prefer SSM port-forwarding."
  type        = string
  default     = ""
}

variable "enable_ssh" {
  description = "Open inbound SSH (22). Off by default; administration is via SSM Session Manager."
  type        = bool
  default     = false
}

variable "ssh_admin_cidr" {
  description = "CIDR allowed to SSH when enable_ssh is true. Never 0.0.0.0/0."
  type        = string
  default     = "127.0.0.1/32"
}

variable "ssh_key_name" {
  description = "Existing EC2 key pair name for SSH. Empty launches with no key pair."
  type        = string
  default     = ""
}

# ── Application ───────────────────────────────────────────────────────────────
variable "app_repo_url" {
  description = "Git URL the instance clones to build and run the docker-compose stack."
  type        = string
  default     = "https://github.com/Carlosmarroquin20/Aegis-RAG.git"
}

variable "ollama_model" {
  description = "Ollama model the stack pulls on first boot."
  type        = string
  default     = "llama3.2"
}

variable "valid_api_keys" {
  description = <<-EOT
    Comma-separated API keys injected into the API. For anything real, do NOT set
    this here (it lands in instance user-data / state) — pull it from SSM Parameter
    Store or Secrets Manager instead. See user_data.sh.tftpl.
  EOT
  type        = string
  default     = "change-me-in-production"
  sensitive   = true
}

variable "extra_tags" {
  description = "Additional tags merged into every resource."
  type        = map(string)
  default     = {}
}
