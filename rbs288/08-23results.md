# 08-23 Results Discussion
## Task 1

For task 1 I tested 4 additional changes to the model.
1. **Addition of extra user and video features**: For video, I added *play_cnt* and *video_duration_x*. For user I added *is_video_author*.
2. **Standardize features**: I standardized the features to have be between 0 and 1 because the ranges varied greatly between different features. 
3. **Removing music_id feature**: I noticed that *music_id* may not be a very useful feature because it's not clear what it indicates, and videos may have many different soundtracks, so I removed this feature.
4. **Changing linear predictor to MLP**: I implemented a simple MLP using PyTorch with 2 layers, a hidden dim of 128, SGD optimizer (lr=0.001, momentum=0.9) and MSE loss. It was trained for 300 epochs.

The results can be seen below. Each modification is built on top of the other, so Mod 2 has the added features from Mod 1, and so on.

| Predictor | Train Loss | Test Loss |
|-----------|------------|-----------|
| Original  | 0.1656     | 0.1595    |
| Mod 1     | 0.1639     | 0.1573    |
| Mod 2     | 0.0974     | 0.0850    |
| Mod 3     | 0.0965     | 0.0842    |
| Mod 4     | 0.0976     | 0.0854    |

1. Adding extra features only improved performance slightly. I tested several combinations of different features to add, and this result was one of the better performers.
2. Standardizing the features helped a lot - I noticed that the range of values for each feature could vary widely so this improvement is understandable.
3. I hypothesized that removing the *music_id* feature might help, and this was supported by the test results. This shows that because the number of features is so low, extraneous features might reduce performance.
4. Surprisingly, the MLP did not perform better than a linear predictor. This may mean that more descriptive features are needed to improve performance further, rather than a more complex model.

## Task 2

For this task, I chose two pairs of groups: content producers vs. non-content producers, and videos above average duration vs. videos below average duration. I thought it would be interesting to compare both user groups and video groups for inequalities.

Content producers were defined as users with either *is_video_author* = 1 or *is_live_streamer* = 1. Approximately 10.6% of the watch instances fall into this category. The other 89.4% were placed in the other group.

The second category was based on the mean video length, which I calculated to be 13005 milliseconds. 16.6% of watch instances were above average video duration, while 83.4% of watch instances were for videos below 13005ms. This shows that the average is brought up by few very long videos, which will likely have implications for the equality metrics later on.

First, I evaluated the loss values of the final MLP predictor on different subgroups.

| Predictor            | Train Loss | Test Loss |
|----------------------|------------|-----------|
| Content Producer     | 0.0877     | 0.0845    |
| Non-content Producer | 0.1007     | 0.1058    |
| Above Avg Duration   | 0.0483     | 0.0487    |
| Below Avg Duration   | 0.0828     | 0.0761    |

There is a clear distinction between performance in both groups. Because one of the features is whether the user is a video producer, I was curious to see if replacing this with one-hot features would improve performance for the highest loss group, non-content producers.

| Predictor             | Train Loss | Test Loss |
|-----------------------|------------|-----------|
| Content Producer      | 0.0877     | 0.0845    |
| Content + one hot     | 0.0912     | 0.0877    |
| Non-content Producer  | 0.1007     | 0.1058    |
| Non-content + one hot | 0.1004     | 0.1054    |

Interestingly, the performance of the content producers was reduced slightly, while the performance of the non-content producers improved slightly. This was not enough to equalize them, but it seems that distinguishing between users who are just live streamers vs video producers is useful to prediction with the content producer subgroup. Also, one-hot features are encrypted to us but are helpful additional information when making predictions for the non-content producer subgroup.

Next, I implemented the classifier and equality metrics using the best performing classifier - linear with added and standardized features.
|                          | Accuracy | Positive Rate | False Positive Rate | False Negative Rate | Positive Predictive Value | Negative Predictive Value |
|--------------------------|----------|---------------|---------------------|---------------------|---------------------------|---------------------------|
| Content Producers        | 0.756    | 0.989         | 0.964               | 0.003               | 0.755                     | 0.818                     |
| Non-Content Producers    | 0.742    | 0.988         | 0.973               | 0.006               | 0.743                     | 0.583                     |
| Above Avg Video Duration | 0.709    | 0.046         | 0.036               | 0.930               | 0.435                     | 0.722                     |
| Below Avg Video Duration | 0.823    | 0.999         | 1.0                 | 0.001               | 0.824                     | 0.0                       |

It is clear that the dataset heavily leans towards positive values, except for videos above the average duration. This makes sense, as longer videos are less likely to be watched all the way. Short videos are so likely to be watched more than halfway that the positive rate is almost 1. There does not seem to be that much variance amongst content producers vs. users who do not produce content, although it seems that content producers are slightly more likely to watch videos. The bias towards a positive prediction thus improves the accuracy of content producers relative to their non-producing counterparts.

**Just as a note**: My commit history is probably a single commit because I messed up forking the repo in the beginning and just moved over my files into this new forked repo. Apologies if anything is a bit messed up as a result, will have it down for next week!