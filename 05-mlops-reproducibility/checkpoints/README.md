# Resume point

If the notebooks are unfinished and the form has to run, use this wiring checkpoint. It does not answer the exercises.

From `05-mlops-reproducibility`:

```sh
make checkpoint
make app
```

`make checkpoint` runs `dvc repro`, then `feast apply` and `feast materialize`. The form loads `fresh-and-good-churn` at alias `champion` and looks up features by customer id.

The skew hunt, the auditor note, and the model race stay unanswered. Ask the instructor for those if you need them.
