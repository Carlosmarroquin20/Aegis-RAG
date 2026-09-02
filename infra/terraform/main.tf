# ─────────────────────────────────────────────────────────────────────────────
# Root module — wires the network and compute modules together and renders the
# cloud-init bootstrap that stands up the Aegis-RAG docker-compose stack.
# ─────────────────────────────────────────────────────────────────────────────

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  tags = merge(var.extra_tags, {
    Project     = var.project_name
    Environment = var.environment
  })

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    app_repo_url   = var.app_repo_url
    ollama_model   = var.ollama_model
    valid_api_keys = var.valid_api_keys
  })
}

# First available AZ in the region — single-AZ by design (cost-conscious).
data "aws_availability_zones" "available" {
  state = "available"
}

# Always resolve the latest Amazon Linux 2023 AMI for the region.
data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

module "network" {
  source = "./modules/network"

  name_prefix         = local.name_prefix
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidr  = var.public_subnet_cidr
  private_subnet_cidr = var.private_subnet_cidr
  availability_zone   = data.aws_availability_zones.available.names[0]
  tags                = local.tags
}

module "compute" {
  source = "./modules/compute"

  name_prefix          = local.name_prefix
  vpc_id               = module.network.vpc_id
  subnet_id            = module.network.public_subnet_id
  ami_id               = data.aws_ssm_parameter.al2023_ami.value
  instance_type        = var.instance_type
  root_volume_gb       = var.root_volume_gb
  user_data            = local.user_data
  api_ingress_cidr     = var.api_ingress_cidr
  grafana_ingress_cidr = var.grafana_ingress_cidr
  enable_ssh           = var.enable_ssh
  ssh_admin_cidr       = var.ssh_admin_cidr
  ssh_key_name         = var.ssh_key_name
  tags                 = local.tags
}
