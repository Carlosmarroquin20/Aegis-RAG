# ─────────────────────────────────────────────────────────────────────────────
# Network module — a minimal, single-AZ VPC for the Aegis-RAG stack.
#
# Cost-conscious by design: there is NO NAT gateway (~$32/mo) and NO load
# balancer. The instance lives in a public subnet and reaches the internet
# directly through the internet gateway. A private subnet is created but left
# unrouted — it is a placeholder for a future data tier (e.g. a managed vector
# store or database) that would sit behind a NAT gateway in a real production
# build. Splitting that out is a documented next step, not a cost we pay here.
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(var.tags, { Name = "${var.name_prefix}-vpc" })
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = merge(var.tags, { Name = "${var.name_prefix}-igw" })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = false # We attach an explicit Elastic IP instead.

  tags = merge(var.tags, { Name = "${var.name_prefix}-public", Tier = "public" })
}

# Reserved for a future private data tier. Intentionally has no route to the
# internet (no NAT gateway) so it costs nothing until it is actually wired up.
resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.this.id
  cidr_block        = var.private_subnet_cidr
  availability_zone = var.availability_zone

  tags = merge(var.tags, { Name = "${var.name_prefix}-private", Tier = "private" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-public-rt" })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}
