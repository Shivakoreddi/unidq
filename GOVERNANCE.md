# UNIDQ Project Governance

## Overview

This document defines the governance model for the UNIDQ (Unified Data Quality) project. UNIDQ is committed to transparent, community-driven development while maintaining high code quality and project sustainability.

## Project Mission

UNIDQ aims to provide a unified, PyTorch-based solution for multi-task data quality assessment, making it easier for data scientists and ML engineers to clean and validate their datasets.

## Roles and Responsibilities

### Maintainers

**Current Maintainers:**
- Shiva Koreddi (@Shivakoreddi) - Core maintainer
- Sravani Sowrupilli (@sravanisowrupilli) - Core maintainer

**Responsibilities:**
- Review and merge pull requests
- Manage releases and versioning
- Triage and respond to issues
- Set project direction and roadmap
- Ensure code quality and test coverage
- Maintain project documentation
- Enforce Code of Conduct
- Ensure PyTorch compatibility with latest releases

**Requirements:**
- Demonstrated expertise in the codebase
- Active contributions over 6+ months
- Commitment to project maintenance (minimum 12 months)
- Understanding of data quality and machine learning principles

### Contributors

Anyone who submits a pull request, opens an issue, or participates in discussions is considered a contributor.

**Recognition:**
- Listed in README.md
- Acknowledged in release notes
- Eligible to become a maintainer based on contributions

### Committers

Contributors who have made significant, sustained contributions may be granted commit access.

**Criteria:**
- 10+ merged pull requests
- 3+ months of active participation
- Demonstrated understanding of project goals
- Agreement to uphold Code of Conduct

## Decision Making

### Minor Changes
- Bug fixes, documentation updates, small features
- **Process:** Single maintainer approval required
- **Timeline:** 2-3 days for review
- **Examples:** Typo fixes, dependency updates, small bug fixes

### Major Changes
- New features, API changes, significant refactoring
- **Process:** 
  1. Open issue for discussion
  2. Consensus from all maintainers required
  3. Community feedback period (7 days minimum)
  4. Implementation with PR review
- **Timeline:** 1-2 weeks
- **Examples:** New data quality tasks, architecture changes

### Breaking Changes
- API breaking changes, major version updates
- **Process:**
  1. RFC (Request for Comments) in GitHub Discussions
  2. Community discussion period (14 days minimum)
  3. Unanimous maintainer approval required
  4. Migration guide must be provided
  5. Deprecation warnings in previous minor version
- **Timeline:** 2-4 weeks minimum
- **Examples:** Removing deprecated APIs, changing core interfaces

## Adding Maintainers

New maintainers are nominated by existing maintainers based on:
- Consistent high-quality contributions (10+ merged PRs)
- Deep understanding of codebase and project goals
- Active participation for 6+ months
- Alignment with project values
- Commitment to long-term maintenance

**Process:**
1. Existing maintainer nominates candidate
2. Private discussion among current maintainers
3. Unanimous approval required
4. Candidate accepts responsibilities
5. Public announcement in repository
6. Update CODEOWNERS, README, and governance documents
7. Grant repository access

## Removing Maintainers

### Voluntary Stepping Down
Maintainers may voluntarily step down at any time by notifying other maintainers.

### Involuntary Removal
Maintainers may be removed for:
- Inactivity (6+ months without contribution or response)
- Repeated Code of Conduct violations
- Consistent neglect of responsibilities
- Actions harmful to the project or community

**Process:**
1. Private discussion among other maintainers
2. Attempt to resolve issues through communication
3. If unresolved, vote to remove (majority of remaining maintainers required)
4. Notify the maintainer being removed
5. Update documentation and revoke access
6. Public announcement if appropriate

### Emeritus Status
Former maintainers in good standing may be granted emeritus status, recognizing their past contributions while clarifying they are no longer active maintainers.

## Release Process

### Version Numbering
Follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Cadence
- **Patch releases**: As needed for critical bug fixes
- **Minor releases**: Monthly or when significant features are ready
- **Major releases**: When necessary for breaking changes

### Release Authority
- Patch releases: Any maintainer can release
- Minor releases: Requires consensus from maintainers
- Major releases: Requires unanimous maintainer approval

### PyTorch Compatibility
UNIDQ commits to supporting the latest two major PyTorch releases at minimum, with updates within 30 days of new PyTorch releases.

See [CONTRIBUTING.md](CONTRIBUTING.md#release-process) for detailed release procedures.

## Communication Channels

### GitHub
- **Issues**: Bug reports, feature requests
- **Discussions**: Questions, ideas, RFCs
- **Pull Requests**: Code contributions

### Email
- **Maintainer contact**: shivacse14@gmail.com, sravani.sowrupilli@gmail.com
- **Private security issues**: Use email for responsible disclosure

### Response Times
- **Critical bugs**: Within 48 hours
- **Security issues**: Within 24 hours
- **General issues/PRs**: Within 7 days
- **Community questions**: Best effort

## Conflict Resolution

1. **Initial discussion**: Attempt to resolve through respectful, good-faith discussion
2. **Mediation**: If unresolved, involve other maintainers as mediators
3. **Maintainer vote**: For technical disputes, maintainers vote (majority decides)
4. **Escalation**: For serious disputes affecting project health, may seek guidance from PyTorch Technical Advisory Council (TAC)

## Project Resources

### Infrastructure
- **Source code**: GitHub (https://github.com/Shivakoreddi/unidq)
- **Package distribution**: PyPI (https://pypi.org/project/unidq/)
- **CI/CD**: GitHub Actions
- **Documentation**: GitHub repository

### Funding and Sponsorship
Currently, UNIDQ is maintained on a volunteer basis. Future sponsorship opportunities will be evaluated transparently with community input.

## Security Policy

Security vulnerabilities should be reported privately to maintainers via email:
- shivacse14@gmail.com
- sravani.sowrupilli@gmail.com

**Do not** report security issues in public GitHub issues.

Maintainers will:
1. Acknowledge receipt within 24 hours
2. Investigate and provide updates within 7 days
3. Coordinate disclosure timeline with reporter
4. Release security patch as quickly as possible
5. Credit reporter (unless anonymity requested)

## Amendments to Governance

This governance document may be amended by:
1. Proposing changes via pull request
2. Discussion period (14 days minimum)
3. Unanimous approval from all active maintainers
4. Update version history below

## Version History

- **v1.0** (December 29, 2025): Initial governance document

## License

All contributions to UNIDQ are licensed under the [MIT License](LICENSE).

## Acknowledgments

This governance framework is inspired by best practices from the PyTorch ecosystem and the Linux Foundation's [Minimum Viable Governance](https://github.com/github/MVG) framework.

---

Last updated: December 29, 2025
