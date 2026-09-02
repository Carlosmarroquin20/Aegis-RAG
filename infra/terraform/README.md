# Aegis-RAG — Terraform (AWS EC2)

Infrastructure-as-code that provisions a single, self-contained AWS environment
running the full Aegis-RAG stack (API + ChromaDB + Ollama + Prometheus + Grafana)
via the project's `docker-compose.yml`.

> ⚠️ **Cost warning.** `terraform apply` creates **billable** resources
> (an EC2 instance, an EBS volume, an Elastic IP). The default `t3.large` is
> **not** free-tier eligible. Nothing here has been applied — review
> `terraform plan` and destroy when you are done. To keep spend at zero, just
> read the code and run `terraform plan`; do not `apply`.

## What it creates

| Resource | Purpose | Notes |
|---|---|---|
| VPC + public subnet + IGW | Network | Single-AZ; a reserved private subnet is created but unrouted |
| Security group | Access control | API (8000) open to a CIDR you choose; SSH off by default |
| IAM role + instance profile | Admin access | `AmazonSSMManagedInstanceCore` → SSM Session Manager, no SSH needed |
| EC2 instance + Elastic IP | Compute | Amazon Linux 2023; boots the docker-compose stack via cloud-init |
| gp3 root volume (encrypted) | Storage | Sized for container images + Ollama models |

**Deliberately omitted to avoid cost/complexity** (documented next steps, not
oversights): NAT gateway, Application Load Balancer + ACM TLS, multi-AZ, managed
data tier, and remote state (see `versions.tf`).

## Prerequisites

- Terraform ≥ 1.6 and AWS credentials (`aws configure` / environment / SSO).
- The [SSM Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
  for shell access without SSH.

## Usage

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # then edit values

terraform init      # downloads the AWS provider (no AWS calls, no cost)
terraform validate  # checks the configuration    (no AWS calls, no cost)
terraform plan      # shows what WOULD be created  (read-only AWS calls)

# Only when you actually want to spend money:
terraform apply

# Access (no SSH):
aws ssm start-session --target "$(terraform output -raw instance_id)"

# Tear everything down:
terraform destroy
```

After `apply`, the instance builds the image and pulls the Ollama model on first
boot — allow several minutes before `terraform output health_url` responds.

## Security notes

- Admin access is via **SSM Session Manager** (audited, no inbound SSH). SSH is
  opt-in (`enable_ssh = true`) and never defaults to `0.0.0.0/0`.
- IMDSv2 is enforced; the root volume is encrypted.
- `api_ingress_cidr` defaults open because the app enforces API-key auth and rate
  limiting — but narrow it to known callers before real exposure.
- **Secrets:** do not put real API keys in `terraform.tfvars`/user-data (both end
  up readable in state / instance metadata). Store them in SSM Parameter Store and
  fetch them in `user_data.sh.tftpl` — the file shows exactly where.

## Layout

```
infra/terraform/
├── versions.tf              # provider + backend constraints
├── providers.tf             # AWS provider + default tags
├── variables.tf             # inputs (region, sizing, CIDRs, …)
├── main.tf                  # AMI/AZ lookups + module wiring
├── outputs.tf               # instance id, API URL, SSM commands
├── user_data.sh.tftpl       # cloud-init: Docker + docker compose up
├── terraform.tfvars.example
└── modules/
    ├── network/             # VPC, subnets, IGW, routing
    └── compute/             # EC2, security group, IAM/SSM, EIP
```
