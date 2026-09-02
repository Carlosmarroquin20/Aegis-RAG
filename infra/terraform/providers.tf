provider "aws" {
  region = var.aws_region

  # Tags applied to every taggable resource, on top of per-resource tags.
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
