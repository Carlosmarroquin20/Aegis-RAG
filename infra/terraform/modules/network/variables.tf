variable "name_prefix" {
  description = "Prefix applied to the Name tag of every network resource."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
}

variable "public_subnet_cidr" {
  description = "CIDR block for the public subnet that hosts the instance."
  type        = string
}

variable "private_subnet_cidr" {
  description = "CIDR block for the reserved (unrouted) private subnet."
  type        = string
}

variable "availability_zone" {
  description = "Availability zone for both subnets (single-AZ by design)."
  type        = string
}

variable "tags" {
  description = "Tags merged into every resource."
  type        = map(string)
  default     = {}
}
