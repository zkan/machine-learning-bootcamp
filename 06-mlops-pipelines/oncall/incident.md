# Incident

**Title.** Churn scores all zero since 03:00.

**Opened.** Today, 03:10.

**What the shop sees.** The form returns a churn probability of 0% for every customer id the team tries. The night operator noticed it on the first three ids, then on ten more.

**What we know.** The pipeline's last green run was yesterday evening. Nobody has deployed a new model since then. The monitor page was not checked before the scores went to zero.

**What we do not know yet.** Whether the registered alias moved, whether the feature table is empty, or whether the form is reading a failed score as zero.
