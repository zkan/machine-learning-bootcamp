# Machine Learning Bootcamp

A six-day course for mixed beginner engineers and business learners. You work on your own laptop. Nothing in this repository calls a cloud service.

This repository currently contains Day 1. Later days are not here yet.

The class is taught in Thai. These notes are in English.

## Syllabus

### [Day 1: Supervised churn classification](01-supervised-churn/)

Build a local churn check for KlongCart, a fictional Bangkok online grocery, from a business question through a Streamlit form.

In this day, you will:

- Set up Python 3.15.2 with mise and uv.
- Read a synthetic customer table and keep the July outcome out of the features.
- Fit logistic regression and a random forest, then compare them with a majority baseline.
- Choose a voucher threshold on the validation rows and score the test rows once.
- Type one customer into a form and read a probability and a yes or no.

Outcome: a classroom model and a form. The score is synthetic. It is not a production retention system.

Setup, the Bangkok timetable, the notebooks, the data, and the form are in that folder. Start with [01-supervised-churn/README.md](01-supervised-churn/README.md).
