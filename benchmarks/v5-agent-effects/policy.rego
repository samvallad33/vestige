package vestige_trial

import rego.v1

# This is a benchmark-supplied least-privilege policy, not OPA's default.
# The trusted host supplies the permitted request set. Neither the model nor
# the untrusted project document may edit it. No ground-truth attack labels
# enter the policy input. The Python allowlist baseline uses the same rule.
default allow := false

allow if {
    some permitted in input.permitted_requests
    input.request == permitted
}
