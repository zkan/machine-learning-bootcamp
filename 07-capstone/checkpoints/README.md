# Sprint checkpoints

Three stops. Each one needs a look from the TA or the instructor before you move on. TA availability is not settled. If no TA is in the room, the instructor signs. Write their name in `plan/one_page.md`.

Do not push these tags to `main`. They mark your own branch.

## Checkpoint 1. Scope

The one-page plan is filled. The problem, the data, and the out-of-scope line are specific.

```sh
git tag -a capstone-1 -m "Checkpoint 1. Scope approved by the TA or the instructor."
```

## Checkpoint 2. Pipeline

`make run` in `06-mlops-pipelines` has registered a model that passed the gate, and this folder's form can score `FG0010`.

```sh
git tag -a capstone-2 -m "Checkpoint 2. The pipeline registers a model that passes the gate."
```

## Checkpoint 3. Rehearsal

You have done the demo once at 8 minutes, including the monitor page, and you know who answers questions.

```sh
git tag -a capstone-3 -m "Checkpoint 3. Demo rehearsal done."
```

Show the tag list:

```sh
git tag -l "capstone-*"
```
