# Security Policy

 - [**Using Whisper Securely**](#using-whisper-securely)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Untrusted environments or networks](#untrusted-environments-or-networks)
   - [Multi-Tenant environments](#multi-tenant-environments)
 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)

## Using Whisper Securely
### Untrusted inputs

Whisper models accept inputs in different audio formats and they're submitted to processes with varying security levels. To prevent that malicious inputs achieve any undesired outcome, you may want to employ the following security measures:

* Sandboxing: Isolate the model process.
* Updates: Keep your model and libraries updated with the latest security patches.
* Input Sanitation: Before feeding data to the model, sanitize inputs. This involves techniques such as:
    * Validation: Enforce strict rules on allowed input formats.
    * Filtering: Remove potentially malicious scripts or code fragments.

### Data privacy

To protect sensitive data from potential leaks or unauthorized access, it is crucial to sandbox the model execution. This means running the model in a secure, isolated environment, which helps mitigate many attack vectors.

### Untrusted environments or networks

If you can't run the models in a secure and isolated environment or if it must be exposed to an untrusted network, make sure to take the following security precautions:
* Confirm the hash of any downloaded artifact (e.g. pre-trained models) matches a known-good value
* Encrypt your data if sending it over the network.

### Multi-Tenant environments

If you intend to run multiple models in parallel with shared memory, it is your responsibility to ensure the models do not interact or access each other's data. The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.

#### Tenant Isolation

Even though we have several tools to help things run smoothly, you must make sure that models run separately. Since models can run code, it's important to use strong isolation methods to prevent unwanted access to the data from other tenants.

Separating networks is also a big part of isolation. If you keep model network traffic separate, you not only prevent unauthorized access to data or models, but also prevent malicious users or tenants sending graphs to execute under another tenant’s identity.

#### Resource Allocation

A denial of service caused by one model can impact the overall system health. Implement safeguards like rate limits, access controls, and health monitoring.

#### Model Sharing

In a multitenant design that allows sharing models, it's crucial to ensure that tenants and users fully understand the potential security risks involved. They must be aware that they will essentially be running code provided by other users. Unfortunately, there are no reliable methods available to detect malicious models, graphs, or checkpoints. To mitigate this risk, the recommended approach is to sandbox the model execution, effectively isolating it from the rest of the system.

#### Hardware Attacks

Besides the virtual environment, the hardware (GPUs or TPUs) can also be attacked. [Research](https://scholar.google.com/scholar?q=gpu+side+channel) has shown that side channel attacks on GPUs are possible, which can make data leak from other models or processes running on the same system at the same time.

## Reporting a Vulnerability

Beware that none of the topics under [Using Whisper Securely](#using-Whisper-securely) are considered vulnerabilities of Whisper.

However, If you have discovered a security vulnerability in this project, please report it privately. **Do not disclose it as a public issue.** We ask for at least 90 days before public exposure as it gives us time to work with you to fix the issue and reduce the chance that the exploit will be used before a patch is released.

Please disclose it as a private [security advisory](https://github.com/openai/whisper/security/advisories/new).
