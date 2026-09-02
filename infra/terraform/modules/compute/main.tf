# ─────────────────────────────────────────────────────────────────────────────
# Compute module — a single EC2 instance that runs the Aegis-RAG docker-compose
# stack, its security group, an SSM-enabled instance profile, and an Elastic IP.
#
# Administrative access defaults to AWS Systems Manager Session Manager (no
# inbound SSH port, no key pair, fully audited). SSH can be enabled explicitly.
# ─────────────────────────────────────────────────────────────────────────────

# ── Security group ────────────────────────────────────────────────────────────
resource "aws_security_group" "this" {
  name        = "${var.name_prefix}-sg"
  description = "Aegis-RAG instance access"
  vpc_id      = var.vpc_id

  tags = merge(var.tags, { Name = "${var.name_prefix}-sg" })
}

# The RAG API. The application enforces API-key auth and rate limiting on top of
# this, but restrict the CIDR here as the first layer of defence.
resource "aws_vpc_security_group_ingress_rule" "api" {
  security_group_id = aws_security_group.this.id
  description       = "Aegis-RAG API"
  cidr_ipv4         = var.api_ingress_cidr
  from_port         = 8000
  to_port           = 8000
  ip_protocol       = "tcp"
}

# Grafana — only opened when a CIDR is supplied. Prefer SSM port-forwarding:
#   aws ssm start-session --target <id> \
#     --document-name AWS-StartPortForwardingSession \
#     --parameters '{"portNumber":["3000"],"localPortNumber":["3000"]}'
resource "aws_vpc_security_group_ingress_rule" "grafana" {
  count             = var.grafana_ingress_cidr == "" ? 0 : 1
  security_group_id = aws_security_group.this.id
  description       = "Grafana dashboard"
  cidr_ipv4         = var.grafana_ingress_cidr
  from_port         = 3000
  to_port           = 3000
  ip_protocol       = "tcp"
}

# Optional SSH. Off by default in favour of SSM Session Manager.
resource "aws_vpc_security_group_ingress_rule" "ssh" {
  count             = var.enable_ssh ? 1 : 0
  security_group_id = aws_security_group.this.id
  description       = "SSH (admin)"
  cidr_ipv4         = var.ssh_admin_cidr
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.this.id
  description       = "Allow all outbound (image/model pulls)"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

# ── IAM: instance profile for SSM Session Manager ─────────────────────────────
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "this" {
  name               = "${var.name_prefix}-instance-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.this.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "this" {
  name = "${var.name_prefix}-instance-profile"
  role = aws_iam_role.this.name
}

# ── EC2 instance ──────────────────────────────────────────────────────────────
resource "aws_instance" "this" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.this.id]
  iam_instance_profile   = aws_iam_instance_profile.this.name
  key_name               = var.ssh_key_name == "" ? null : var.ssh_key_name
  user_data              = var.user_data

  metadata_options {
    http_tokens   = "required" # Enforce IMDSv2.
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-instance" })
}

resource "aws_eip" "this" {
  instance = aws_instance.this.id
  domain   = "vpc"

  tags = merge(var.tags, { Name = "${var.name_prefix}-eip" })
}
