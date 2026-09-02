terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Remote state is recommended for anything beyond a local experiment. Create
  # the bucket + lock table once (out of band), then uncomment and `terraform
  # init -migrate-state`. Left commented so the skeleton works with local state.
  #
  # backend "s3" {
  #   bucket         = "aegis-rag-tfstate-<account-id>"
  #   key            = "aegis-rag/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "aegis-rag-tflock"
  #   encrypt        = true
  # }
}
