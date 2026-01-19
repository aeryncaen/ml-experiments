# Data Sources for Training

## Readily Available (No Credentials)

### 1. Gitleaks (Secrets - Positive)
- **Repo:** https://github.com/gitleaks/gitleaks
- **Location:** `cmd/generate/config/rules/*.go`
- **Format:** Go functions calling `utils.GenerateSampleSecrets("identifier", "secret")`
- **Example:**
  ```go
  tps := utils.GenerateSampleSecrets("AWS", "AKIALALEMEL33243OLIB")
  ```

### 2. detect-secrets (Secrets - Positive)
- **Repo:** https://github.com/Yelp/detect-secrets
- **Locations:**
  - `tests/plugins/*_test.py` - Module-level constants like `EXAMPLE_SECRET = '...'`
  - `test_data/each_secret.py` - One example of each secret type
  - `test_data/config.yaml`, `config.ini`, etc. - Various formats
- **Example:**
  ```python
  EXAMPLE_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
  ```

### 3. TruffleHog (Secrets - Positive, Unit Tests Only)
- **Repo:** https://github.com/trufflesecurity/trufflehog
- **Location:** `pkg/detectors/*/*_test.go` (unit tests, NOT integration tests)
- **Format:** Go string literals in test cases
- **Example:**
  ```go
  validPattern = `[{"test_secrets": {"github_secret": "ghs_RWGUZ6..."}}]`
  ```
- **Note:** Integration tests use GCP Secret Manager - skip those.

### 4. Generated False Positives
- UUIDs: `550e8400-e29b-41d4-a716-446655440000`
- MD5: `d41d8cd98f00b204e9800998ecf8427e`
- SHA1: `da39a3ee5e6b4b0d3255bfef95601890afd80709`
- SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Base64: Random bytes encoded

## Requires Special Access (Later)
- FPSecretBench: 1.5M false positives (requires DPA)
- SecretBench: 15K secrets with context (requires DPA)
