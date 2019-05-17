CNNs. Notes

### Different types of filters in CNNs

* Right Sobel Filter
* Top Sobel Filter
* Blur Filter



Image|13x8
---|---
Filter| 3x3
Filter Map| 11x6

Output Height = (Input Height - Filter Height )/ (Row Stride )+1   
Output Width = (Input Width - Filter Width )/ (Row Stride )+1

Filter has same number chanels are the Image

Image |Filter |Feature Map
---|---|---|---
720 x 960 x 3 | 3 x 3 x 3 | 718 x 958


## Padding in CNN

Valid
Same

Convolutional Layers: To pad or not to pad?
There are couple of reasons padding is important:

It's easier to design networks if we preserve the height and width and don't have to worry too much about tensor dimensions when going from one layer to another because dimensions will just "work".
It allows us to design deeper networks. Without padding, reduction in volume size would reduce too quickly.
Padding actually improves performance by keeping information at the borders.
Quote from Stanford lectures: "In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly." - source


https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad



## Pooling
Used for demensionality reduction

__Max Pooling__ take max value of a from a pool of pixels.   
__Average Pooling__ take average value of a from a pool of pixels.
Give 3d output for a 3d image

#### Special type of pooling
Give 1d output for a 3d input.
___Global Average Pooling___ Take average of each chanel and get an average of the layer.

___Global Max Pooling___ Take average of each chanel and get an average of the layer.


Questions
1. Which of the following layer is a downsampling layer?
2. Pooling helps in reducing the dimension of the input image.
3. What are the different types of pooling that can be applied to a layer?
4. Consider the below 2-D Input Image of shape (4 X 4):


 We have applied Average Pooling on the above image with stride = 2. What will be the Output Image?
5. Which of the following pooling strategy can be used to convert a 3-D image to 1-D image?


### Architecture of CNN using Filters, Padding and Pooling Layers


What is the correct sequence of a network with CNN architecture?

Input -> CNN -> Fully COnnected Neural Network -> Output

Which of the following is/are hyperparameter(s) for a Convolutional Neural Network?

Choose only ONE best answer.

A Number of Convolutional Filters
B Size of Pooling
C Stride
D Number of Pooling Filters
E All of the above

https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/

# Mentorship best practices

A resource of best practices and tips for SharpestMinds mentorships.  If you're a mentor and you have an idea for something to add, please submit a PR! ðŸ™‚

## Interview guide

When you interview a mentee applicant, it helps to set expectations. Here's an example of a format for a first conversation that tends to work well:

1. One hour long.
2. For the first 15 minutes or so, let the applicant tell you about their background.
3. For the next 15 minutes or so, give a quick overview about what you work on, what your expertise is, and how you'd structure the mentorship. You can use the template (in the chat channel with the mentee) as an example.
4. For the last 30 minutes, go into the details of what the applicant's general strengths and weaknesses are. If they seem promising, you can also discuss possible projects with them.

Here are some good general questions to ask a mentee applicant:

 - What's your schedule like and how much time can you commit?
 - What's your educational background? Which MOOCs have you completed?
 - Have you done any other data science projects? Can you walk me through one?
 - Do you know SQL?
 - Do you know common ML algorithm X? (where X = SVD, t-SNE, backprop, ... etc.)
 - How comfortable would you be explaining common algorithm X to me right now? What if I gave you 15 minutes to review X? What if I gave you overnight to review X?
 - Have you done any problems on LeetCode or HackerRank?
 - Have you done anything with servers? (e.g., Flask, Django, AWS, GCP)
 - Do you have any projects or problems in mind that you'd like to work on?
 - What industries are you interested in working in?
 - What's your ideal company to work at? What's your ideal role?
 - Where have you applied already?


**NOTE:** At this stage, a mentee most likely doesn't know what a good project idea looks like. That's all right: they aren't in the industry yet, and you have veto power over any project they propose.

You don't need to have decided on a project by the time you've started the mentorship, but if not, **you should make this the goal of your first week.**

## On your first call

A checklist of things to do on your first call with a mentee:

1. **Schedule weekly or bi-weekly meetings:**
 - Pick a time and place. (e.g. Google Hangouts, Wednesdays 6pm)
 - Both of you will probably have to reschedule some weeks, so make sure to set some guidelines around this. (e.g. let each other know 24 hours ahead of time if you need to reschedule)


2. **Agree on the meeting format:**
 - *Suggestion:* Review last week's goals, set new goals for week.
 - *Suggestion:* During the week, it helps a lot to keep a running list of non-urgent issues/questions/things to bring up during meeting. Both mentee and mentor should do this. Otherwise, something important might occur to you during the week, but by meeting time you'll have forgotten what it was.


3. **Set expectations:**
 - You and your mentee both share the same goal, so honesty is everything. Be honest about what mentees should expect from you, and what you expect from them.
 - What kind of response time and availability should they expect from you? What kind of response time and availabiilty should you expect from them?
 - Tell mentees to be honest if they feel like they are not getting enough value or are unsatisfied with the mentorship. If this ever happens, let us know by sending an admin a DM on Slack. We can support you if any issues come up.
 - What is the primary communication channel you'll use? (eg. email, Slack, SharpestMinds)


4. **Create a rough plan with a timeline:**
 - This will probably evolve a lot in the first couple of weeks as you both get comfortable and settle on a well-defined project.
 - If you haven't decided on a project by the time you've started, **make this the goal of the first week.** Point the mentee towards some resources or possible datasets, or suggest your own project ideas. A dataset shouldn't be pre-cleaned, but ideally should be something the mentee gathers themselves through web scraping or direct collection.
 - **TODO**: link to example projects from other SharpestMinds mentorships ðŸ—ï¸
 - Projects should loosely follow these steps:
 - Data collection:
	- Data analysis/exploration
	- Feature engineering/selection
	- Model building
	- Model deployment
 - *Suggestion*: Set up a GitHub repo, and have your mentee submit PRs for your approval.
 - *Suggestion*: Encourage your mentee to write blog posts about significant milestones.
 - **TODO**: Link to some resources on structuring a data science project ðŸ—ï¸
 - **Protip:** Try and structure your mentee's workflow like you would in an industry setting. (e.g. agile/scrum, CI/CD tooling)
 - **TODO**: list useful tools for remote collaboration ðŸ—ï¸


5. **Connect with your mentee on your networks:**
 - Connect on LinkedIn, Twitter, etc.
 - If you're in the same city as them, give them details on useful meetups/events/groups. SharpestMinds has info on this for many cities, so reach out to us if you'd like advice.
 - Ask your mentee what their ideal company and role is. Think about any of your connections that you might be able to connect them with.
 - **NOTE:** Many new grads are pretty ignorant about the possible roles out there and won't have a great idea of what they are looking for. This is okay; they just need more exposure. Setting up informational interviews with people in your network can be very helpful to start giving them a birds-eye view of the industry.
 - If your mentee (and you) build something cool, let us know! We have the ability to share to a very wide audience: SharpestMinds content gets well over **a quarter million** monthly views across all platforms. ðŸ‘€
 - Encourage mentees to attend SharpestMinds group office hours and connect with other mentees. This
is one of the highest value activities they can do.
