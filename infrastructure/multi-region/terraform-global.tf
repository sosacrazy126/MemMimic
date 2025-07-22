# MemMimic Multi-Region Infrastructure
# Global resources for disaster recovery and high availability

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = "memmimic-terraform-state"
    key    = "global/terraform.tfstate"
    region = "us-west-2"
    
    dynamodb_table = "memmimic-terraform-locks"
    encrypt        = true
  }
}

# Configure AWS providers for multiple regions
provider "aws" {
  alias  = "primary"
  region = var.primary_region
}

provider "aws" {
  alias  = "secondary"
  region = var.secondary_region
}

provider "aws" {
  alias  = "tertiary"
  region = var.tertiary_region
}

# Variables
variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-west-2"
}

variable "secondary_region" {
  description = "Secondary AWS region for DR"
  type        = string
  default     = "us-east-1"
}

variable "tertiary_region" {
  description = "Tertiary AWS region for global distribution"
  type        = string
  default     = "eu-west-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "memmimic"
}

# Global S3 Bucket for Backups with Cross-Region Replication
resource "aws_s3_bucket" "backup_primary" {
  provider = aws.primary
  bucket   = "${var.project_name}-backups-${var.primary_region}"

  tags = {
    Name        = "${var.project_name}-backups-primary"
    Environment = var.environment
    Purpose     = "backup-storage"
  }
}

resource "aws_s3_bucket" "backup_secondary" {
  provider = aws.secondary
  bucket   = "${var.project_name}-backups-${var.secondary_region}"

  tags = {
    Name        = "${var.project_name}-backups-secondary"
    Environment = var.environment
    Purpose     = "backup-storage-replica"
  }
}

resource "aws_s3_bucket_versioning" "backup_primary" {
  provider = aws.primary
  bucket   = aws_s3_bucket.backup_primary.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "backup_secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.backup_secondary.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_replication_configuration" "backup_replication" {
  provider   = aws.primary
  depends_on = [aws_s3_bucket_versioning.backup_primary]

  role   = aws_iam_role.replication.arn
  bucket = aws_s3_bucket.backup_primary.id

  rule {
    id     = "backup-replication"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.backup_secondary.arn
      storage_class = "STANDARD_IA"
      
      replica_kms_key_id = aws_kms_key.backup_secondary.arn
    }
  }
}

# KMS Keys for encryption in each region
resource "aws_kms_key" "backup_primary" {
  provider                = aws.primary
  description             = "KMS key for MemMimic backups in primary region"
  deletion_window_in_days = 7

  tags = {
    Name        = "${var.project_name}-backup-key-primary"
    Environment = var.environment
  }
}

resource "aws_kms_key" "backup_secondary" {
  provider                = aws.secondary
  description             = "KMS key for MemMimic backups in secondary region"
  deletion_window_in_days = 7

  tags = {
    Name        = "${var.project_name}-backup-key-secondary"
    Environment = var.environment
  }
}

# IAM Role for S3 Replication
resource "aws_iam_role" "replication" {
  provider = aws.primary
  name     = "${var.project_name}-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-s3-replication-role"
    Environment = var.environment
  }
}

resource "aws_iam_policy" "replication" {
  provider = aws.primary
  name     = "${var.project_name}-s3-replication-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl"
        ]
        Resource = "${aws_s3_bucket.backup_primary.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.backup_primary.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete"
        ]
        Resource = "${aws_s3_bucket.backup_secondary.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.backup_primary.arn
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.backup_secondary.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "replication" {
  provider   = aws.primary
  role       = aws_iam_role.replication.name
  policy_arn = aws_iam_policy.replication.arn
}

# Route53 Hosted Zone for Global DNS
resource "aws_route53_zone" "main" {
  provider = aws.primary
  name     = "${var.project_name}.com"

  tags = {
    Name        = "${var.project_name}-dns-zone"
    Environment = var.environment
  }
}

# Global CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  provider = aws.primary
  
  origin {
    domain_name = "api.${var.project_name}.com"
    origin_id   = "primary-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  origin {
    domain_name = "api-${var.secondary_region}.${var.project_name}.com"
    origin_id   = "secondary-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"

  aliases = ["${var.project_name}.com", "www.${var.project_name}.com"]

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "primary-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "CloudFront-Forwarded-Proto"]
      
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 86400
  }

  # Health check for failover
  ordered_cache_behavior {
    path_pattern     = "/health"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "primary-origin"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl                = 0
    default_ttl            = 30
    max_ttl                = 60
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.main.arn
    minimum_protocol_version = "TLSv1.2_2021"
    ssl_support_method       = "sni-only"
  }

  custom_error_response {
    error_code         = 503
    response_code      = 503
    response_page_path = "/maintenance.html"
  }

  tags = {
    Name        = "${var.project_name}-cloudfront"
    Environment = var.environment
  }
}

# ACM Certificate for HTTPS
resource "aws_acm_certificate" "main" {
  provider                  = aws.primary
  domain_name               = "${var.project_name}.com"
  subject_alternative_names = ["*.${var.project_name}.com"]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.project_name}-certificate"
    Environment = var.environment
  }
}

resource "aws_acm_certificate_validation" "main" {
  provider                = aws.primary
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

resource "aws_route53_record" "cert_validation" {
  provider = aws.primary
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main.zone_id
}

# RDS Cross-Region Automated Backups
resource "aws_db_subnet_group" "primary" {
  provider = aws.primary
  name     = "${var.project_name}-db-subnet-group-primary"
  
  subnet_ids = data.aws_subnets.primary.ids

  tags = {
    Name        = "${var.project_name}-db-subnet-group-primary"
    Environment = var.environment
  }
}

resource "aws_db_subnet_group" "secondary" {
  provider = aws.secondary
  name     = "${var.project_name}-db-subnet-group-secondary"
  
  subnet_ids = data.aws_subnets.secondary.ids

  tags = {
    Name        = "${var.project_name}-db-subnet-group-secondary"
    Environment = var.environment
  }
}

# Data sources for existing VPCs and subnets
data "aws_vpc" "primary" {
  provider = aws.primary
  tags = {
    Name = "${var.project_name}-vpc-primary"
  }
}

data "aws_vpc" "secondary" {
  provider = aws.secondary
  tags = {
    Name = "${var.project_name}-vpc-secondary"
  }
}

data "aws_subnets" "primary" {
  provider = aws.primary
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.primary.id]
  }
  tags = {
    Type = "private"
  }
}

data "aws_subnets" "secondary" {
  provider = aws.secondary
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.secondary.id]
  }
  tags = {
    Type = "private"
  }
}

# Outputs
output "backup_bucket_primary" {
  description = "Primary backup bucket name"
  value       = aws_s3_bucket.backup_primary.bucket
}

output "backup_bucket_secondary" {
  description = "Secondary backup bucket name"
  value       = aws_s3_bucket.backup_secondary.bucket
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.main.id
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.main.domain_name
}

output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = aws_route53_zone.main.zone_id
}

output "certificate_arn" {
  description = "ACM certificate ARN"
  value       = aws_acm_certificate.main.arn
}