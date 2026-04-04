# Secure Python Guidance Summary

- Validate inputs before using them in control flow or data access.
- Limit trust in dynamic data and avoid unsafe evaluation patterns.
- Prefer explicit error handling over broad exception suppression.
- Do not log secrets or sensitive values.
- Keep data transformations simple and auditable.
- Structure code so security checks are easy to review and test.
