# Configuration

`default.yaml` contains safe local defaults for the educational retrieval system.

Configuration precedence is:

1. Command-line arguments
2. Environment variables
3. YAML configuration
4. Application defaults

Use `.env.example` as the template for local environment variables. Do not store private paths,
credentials, or dataset access tokens in committed configuration files.
