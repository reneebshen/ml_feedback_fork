# Results Discussion
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

The second category was based on the mean video length, which I calculated to be 13005 milliseconds. 16.6% of watch instances were below average video duration, while 83.4% of watch instances were for videos above 13005ms.

First, I evaluated the loss values of the final MLP predictor on different subgroups.

| Predictor            | Train Loss | Test Loss |
|----------------------|------------|-----------|
| Content Producer     | 0.0877     | 0.0845    |
| Non-content Producer | 0.1007     | 0.1058    |
| Below Avg Duration   | 0.0483     | 0.0487    |
| Above Avg Duration   | 0.0828     | 0.0761    |