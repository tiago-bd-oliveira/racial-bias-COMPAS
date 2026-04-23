# Ethical Considerations

Ethics is a big part of this project. Given the fact that our study aims to bring racial bias to light, it's important that we follow very strict ethical practices.

## Ethics in Feature Selection

In terms of features for the training of our first model, we focused purely on the defendants personal features and criminal record. We did not take into account any information that is not a personal characteristic of the defendant. 

We also had to take into account that the COMPAS decile score is calculated before a sentence is given. Because of this, we did not take into account any information linked to events that happen after the COMPAS screening where the score was attributed to the subject. 

Here are some examples of those decisions:

- **Exclude jail time information**: As mentioned before, the score is calculated before the sentence is given, to help the judge make a better decision. So it is not a cause for the score, instead it is a consequence of it.

- **Not measuring arrest time**: Since we had both the offense date and the arrest date, we could have included the time it took for the subject to get arrested. However, we decided **not to** because that would have more to do with police efficiency than the actual danger of the person.

- **Derive more criminal record information**: We decided to derive a lot of metrics from the original criminal statistics, to test some bias hypothesis. It is easier to prove the bias if correlation to the final score is bigger for the race variable than for the criminal flags like `velocity_repeat_offender`.

## Scope of our Ethics

This project aims to expose the racial bias of the COMPAS score. However the score is only calculated when the subject is arrested. There is already a level of bias on the decision to arrest someone, by part of the police. What we are trying to say is that it is hard to mitigate the bias when the existence of the records in itself could happen due to bias.

However this is something we cannot control, and is totally out of our scope. We fully accept the risk that bias may persist simply through the fact that some criminals don't get arrested, and some people do unjustly.