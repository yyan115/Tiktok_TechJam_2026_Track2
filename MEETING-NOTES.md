Created 2026-08-26 4:06PM

The webinar has been held and more information has been received. Here is the transcript of the meeting, along with questions asked at the end. Please piece together the information yourself, where and which questions were asked, if it was answered or not, etc. The transcript is not perfect, if it is not reliable, just forget about it instead.

In addition, the slides are provided as well, they are ordered and may not be in the correct order but it should be.

---

So do I need to, so do I need to open my camera?
Yeah.
Thank you. Perfect.
Does it work?
Yes. Perfect. Looks great. Thank you so much.
Okay. Thank you.
Cool. Cool.
And welcome to our second TikTok Tech Jam 2026 Technical Workshop,
where we will be diving into track two,
Autonomous Machine Learning Research Agent for Recommended Systems.
So before we begin, I'm Phyllis, and it's very nice to meet you.
I'm part of the APEC Early Careers team here at TikTok,
and I've been with the team for more than four and a half years now,
and I manage regional university relations
and campus engagements in Southeast Asia and Japan.
I'm also the project manager of TikTok Tech Jam 2026,
and I'm excited to have you join us today at this webinar
where we would actually explore more and understand track two a little better.
So for today's agenda,
Haichang Zhou, our engineer who built track two,
would actually kick us off with his insights to help us to better appreciate
and understand the team behind this problem,
some technical background, and then deep dive into the problem statement
and tips to help you be successful in this hackathon.
We will then have a very short Q&A with Haichang,
where you are able to type your questions in our chat box,
and Haichang will be answering them one at a time.
I'll then end off the webinar with a recap of Tech Jam,
and you can also feel free to ask any questions you have about the hackathon then.
So just a very quick overview.
TikTok Tech Jam is our annual flagship student hackathon,
and we're guided by our hackathon mission to build with joy and code for change.
So build with joy is a celebration of learning, collaboration, and curiosity
in line with TikTok's mission to inspire creativity and bring joy.
Participants are encouraged to embrace the spirit of innovation
where you can experiment freely, support one another,
and just have a lot of fun along the way
as you grow together as builders of the future.
Now code for change challenges participants to learn,
to actually think beyond the code and focus on impact.
We hope that through this hackathon, teams are empowered to build solutions
that actually drive positive change,
solve real-world business problems,
and also reflect TikTok's belief in shaping the future
with responsible technology.
So if you haven't already checked out the other problem statements,
they are all published on our information document,
because this year you can either fly solo
or participate in teams of up to five members
to solve problems across five distinct tracks.
And did I mention, this year we have an even bigger prize pool
with the first place at Singapore Dollars $15,000,
followed by second place $8,000,
third place $5,000,
and fourth and fifth place at $3,000.
We also have a special People's Choice Award,
and the People's Choice Award winner is selected
via public voting on DevPost.
So do rally all your family and friends
to show your project a little love and support,
and voting is actually from 1st September 3pm to 7th September 3pm.
So without further ado,
I would like to pass the time now to Haicheng,
who actually deep dives into the problem statement
and to share a little bit more about Track 2.
Haicheng, please.
Okay, thank you.
So let's get started.
The Track 2 is
Atomina's machine learning research agent
for commander assistant.
So let me first briefly introduce
the team I'm working for.
This is the team we together built this problem.
And we are the content recommendation
algorithm team in TikTok Shop.
So maybe some of you have already used TikTok,
and actually TikTok has provided
various different types of videos and lives.
And one of the videos and lives
are e-commerce videos and e-commerce livestreams.
So the goal of our team is to recommend
different e-commerce content
from billions of candidates to each user.
So this is why we prepare
a recommender assistant problem
at this track.
So let me first briefly introduce
the work of our team.
So actually this pipeline is very similar
to our daily work.
First, in this track,
you need to read the problem.
It's very similar to our daily situation.
Then we need to understand
the everyday data and the target metrics.
And then you need to inspect the data distribution
through some exploratory data analysis.
Like maybe you use some SQL to check the,
for example, some samples of our data
to get a rough understanding
of what the data really looks like.
And then you need to build some features.
You need to build and select
features from the data set.
Then you can use this feature
and build a model.
And train the model,
select the loss function
and choose the hyperparameters.
Then you evaluate the model
using the matrix and check
whether the model is overfitting or underfitting.
And then you reflect and revise your data.
And you can then take this loop for another time.
This is a loop.
And through this loop,
you will make your model better and better.
Actually this is quite similar to our daily work.
And then we can finally build this recommender system
more accurate.
And finally, each user will get the videos
or lives that they would prefer.
So let me briefly introduce the background
of the recommender system.
Actually I graduated from NTU one year ago.
And I know that the university does not teach
industry-style recommender system.
So I think if I can briefly introduce
the industry-style recommender system
to the students,
you will get more familiar with
what the recommender system works in a company.
So actually the industry-style recommender system
is like a funnel.
We have a multi-stage funnel.
And in each funnel,
the candidate videos or the candidate live stream
will get more accurate.
And finally, we give the very few
but very accurate videos or live streams to the user.
So first, we will use a recall stage or retrieval stage.
In this stage, we will check from
millions of candidate videos.
And then after this stage,
we will get thousands or hundreds of candidate videos.
And then we use a pre-rank,
following a ranking stage.
And after the stage,
we will get like tons of items, candidates.
And finally, we use a re-rank stage
to generate like around 10 videos
or live streams to the user.
And this list is displayed to the users.
This list is what you really see in the TikTok app.
So you may ask,
why an industry-style recommender system is cascade-style?
Why is it so complicated?
Actually, this is because
we want to make this recommender system more great.
So actually, a very vanilla idea is that
we use one ranking model to score all the candidate content,
maybe millions of candidate content.
And then we get a very accurate ranking
and select the top 10 or top 20 videos to the user.
But actually, a very accurate ranking model,
using a very accurate ranking model
to all the candidate content is very expensive.
You need a lot of time,
and you need a lot of computational resources to get the score.
You may ask,
is it compulsory that we need to score all the candidates very accurate?
Can we use other, more faster methods?
This is why we use a cascaded stage.
First, we use a record stage,
which selects the relevant thousands of items
or candidates from all the videos.
Then we use a ranking model here
to select the top tons of candidates.
And then we use a re-ranking model to get the final order.
The pre-rank stage is just to...
Actually, a pre-ranking model is a light ranking model.
Because sometimes it is still very expensive for a ranking model
to score several thousands or several hundreds of videos.
So we need a pre-ranking model to pre-score
and select maybe the top 100 or 200 items to the ranking model.
And we use the ranking model,
the most complicated and most accurate model,
to give the accurate score
and select the top tons of candidates' videos.
So this is why the industry-style recommender system
is designed as a cascaded model.
So let me first briefly introduce the stage called recall or retrieval.
In this stage, the recommender system will need to select
maybe thousands, several thousands or several hundreds of videos
from millions of videos.
And in this stage, we do not really care
the very accurate ranking score.
Because we have following accurate ranking model.
In this stage, we just need to get some very relevant candidates.
And we do not care about their ordering.
Okay, let me just skip the pre-ranking stage,
because this stage is just a small ranking model.
And then we have maybe several hundreds of candidate videos.
We use a ranking model,
a very complex and very accurate ranking model
to give a very accurate score to each of the video.
And then we order the video according to the score.
Here we also explain why the ranking model is very heavy.
Because our goal is to get a very accurate score.
And because the input is small,
because we have only like 100 or 200 videos here to score.
So we can use a very complex and very big model.
And then in this stage,
so how to design a very accurate model,
we need to extract and design rich features,
like the user's behavior history,
and the features of videos or the product,
and some context.
And then this is where the feature engineering pays off.
Actually, how to design the feature and preprocess the feature
is very important for the performance of the ranking model.
And this stage is also for this challenge.
So in this challenge,
you need to score each candidate and then evaluate the ordering.
So you can see the English figure that we provide.
In this stage,
the recommender system can process multiple types of features
and use a large ranking model,
get the score and order the videos according to the score.
And finally, this is the re-ranking stage.
Actually, this stage is not about our challenge,
but actually this stage is also very important
for a recommender system.
You can regard it as a background
for you to understand the whole recommender system.
So why we need to, in the previous stage,
in the ranking stage,
we already get a very accurate score and ordering
of each of the video or the product.
So you may ask, why we still need a re-ranking stage?
Let me first introduce how you get a list of videos in TikTok app.
Actually, the recommender system finally gets like maybe
a list of 10 or 20 videos or live sales in a list.
And then the server sends the list to your TikTok app.
And actually, the output of the recommender system is the list.
So let me imagine what is the videos in the list are ordered
according to the interest.
Like that the first video is your favorite,
and the second is your second favorite.
So do you think this order is very good,
or is there still some issues?
Actually, if the videos are ordered according to this,
you will find that the first one or two videos are your favorite.
And then you will find the videos get more boring,
or you do not want to get to the next one.
So we just stop here, and you will not go to the next video.
So this is that if you order the final list according to the ranking order,
so you will only watch the first or the second or third videos,
and you will not go to the next.
But this is not what we like.
We want you to watch more videos and view more products.
So we need to re-rank the order.
This is why we need a re-ranking stage.
So we claim that actually the output of the recommender system is the list.
We want the users like the list of videos, not each item, not each video.
So the ranking stage is that it reorders the videos,
and the users may like the list,
and they will finish watching the 10 or 20 videos in the list,
and they would like to get the next list.
Here we give an example.
Like in the left part, the videos are ordered according to the ranking score.
So the first video is your favorite, and the second is your second favorite.
You will be very happy watching the first one or the second one,
and you will get less happy, and you may stop at the third video,
and you will not watch the following.
But if we reorder the videos according to the right part,
you will feel very happy watching the first, and you will also watch the second,
and in the third, you will find, oh, this video is not very interesting.
You will want to try to watch the next, and you go to the next,
and you find that, oh, this video is also very interesting,
and you will watch this one and go to the next,
and you will go to the next one, the next, and the next.
So this is why the re-ranking stage is designed.
So previously I just introduced why the recommender system is designed as the four stages.
Then I will introduce where the data comes from.
Previously in the ranking model,
I have shown that we have various features like the user profile,
like what you may provide to the TikTok app,
the behavior history, like you may click like or click dislike to a video.
The item features, the features come from the video itself
or the product linked to the video, and some context.
So here it introduces the actions or the labels of your actions.
So one user was shown with various videos,
and you may click on this one, not click the other one,
and you may like or dislike or comment on some.
So here we will record the user ID, the video, the context,
like maybe you watch the video in the evening,
and the label that you click this one, you comment or you dislike.
So all these actions will be recorded,
and those actions are very important to train a very accurate recommender system.
But you may see that only the video that's shown to the user has those labels.
Then I will briefly introduce the basic matrix in recommender system,
like CTR and CVR.
The CTR means the click and condition of the impression.
You can see it directing this picture.
The CTR is calculated as 40,000 over 1 million.
And the CVR, CV means conversion, this is the conversion over click.
So this can be computed as 1,200 over 40,000.
So the two matrix are chained.
First, a video has an impression, and then some of the impressions has click,
and among the clicks there are CVR conversion.
So each step keeps a small fraction.
So positive gets much real further down.
Hi Phyllis, will you provide the transcript to the students
just so someone has a question?
Yeah, we will.
The links to the recording will be shared by Tamara and Noonuki.
We can address the questions later.
Haichang, you just go ahead first.
Okay.
Then I will briefly introduce how a ranking model is scored.
So here we need a metric to score the CTR and CVR or some other metrics.
Here you can see that the accuracy of Model A is 96%.
And it's AUC maybe only 1.5.
And Model B, the accuracy of Model B is 91%.
And AUC maybe 0.78.
It's like that the Model A always predicts no click,
because only 4% of the video impressions are clicked.
But we cannot say this model is very good,
because this model only predicts a no click.
It does not provide any information.
So here we can see that the metric of accuracy does not work.
We need the AUC score to score how the model works.
Okay.
I think I have briefly introduced...
Okay, here I will briefly introduce what a feature looks like.
We have several kinds of features.
The most important features are categorical IDs like the user ID,
the product item ID, the category ID, and some others.
And the second is that canary of the point.
It's like that, for example,
item ID column can hold millions of distinct values.
Here we see why it's not deep.
For example, we may have millions of item IDs,
but for each video or each item in each prediction,
we only select one.
So we need an embedding table.
And then in each prediction, we only take one item ID.
Okay, so let's go to the problem statement.
This is more about the task itself.
So you can see that in the previous...
Previously, I have briefly introduced how recommender system works
and what is the daily works of our machine learning engineers
in recommender system.
So you can see that some of the works can be automated.
So this is why we designed this track,
and we asked you to use agent to accelerate your model updates.
So here the agent can see the data split and the evaluations,
and then the agent can take loops to reproduce iterate
and improve the metric score validation.
And finally, we need you to submit your answer.
Actually, here we have already provided the test set to you.
But I don't suggest you to use the test set to improve your model.
Because according to our daily works,
we find that once you use some data from your test set,
your model's real performance will drop dramatically in the future.
In my work, one day I just use a test set,
one day data from a test,
and in the following days I have checked that the AUC of my model drops from 10 points.
So this is very, very critical that you do not use the data from test set.
So in this stage, you can first reproduce the baseline.
Actually, I have updated the starter toolkit.
There is a very simple model in this starter toolkit,
and we have also provided the metrics, the column metrics.
You can start off working an end-to-end pipeline
and confirm it reaches the validation score reported by the organizer provided baseline.
And given the baseline, you can use the agent to start your iteration.
And during the iteration, you can improve over the baseline.
Okay, I think this stage has already provided in the docs.
So here I would like to also provide some tips for the hacksaw.
In this track, we asked you to use agent to improve the performance of the model.
You can use whatever AI agent you like,
but if you do not, you need to use more tokens,
and you do not have a subscribed version of GBT or cloud code.
You can also try the AI tool from Bydance.
Here the company provides a 7-day free trial of this version.
You can try to use it.
Okay, I think I have also introduced this stage.
Okay, so let's get to the QA session.
Yeah, thanks so much, Haitang.
So if you guys have questions, feel free to share them into the chat.
I think we're just waiting for more questions.
Anything before this is from the previous session.
If anyone has any questions, feel free to write them in the chat below.
Okay, thank you.
Haitang, I think we have our first question.
Do we need to submit a video for this track?
So, Felix, I also want to ask you,
do you need them to submit the videos?
It depends on the track owner.
So actually all the other tracks, if I'm not wrong,
if I remember correctly, all the other four actually require a video.
Yeah, because they want to see the demo on video.
But then for your track, is that something we would like to implement?
Okay, okay, okay.
It's also okay for them to submit a video for the track.
Okay, so Haitang, if let's say they don't want to submit a video,
is it okay with you?
It's also okay not to submit a video.
If you do not submit a video, you will need to write your report carefully
because usually a video may provide more information than a report.
So maybe your report will contain more information for us to carefully take.
Consider your solution.
It depends on you.
Whether you like to write a longer report
or want to have a record to give more details.
Okay, got it.
Maybe in the problem statement later,
I will write a quick update that a video is not compulsory.
However, video tends to be able to show more.
If not, then make sure that your report is longer and more detailed.
Yeah, yeah, yeah.
Thank you.
This is very helpful.
And that's a great question.
Thank you.
Thank you.
Okay, we do have quite a few questions.
Maybe a quick one.
Will we share the recording?
Yes.
All the webinars today will be recorded.
And then the links will be provided in the information document.
But this will only be provided by tomorrow, 12 p.m.
Okay, thank you.
So maybe the next question from Rohan.
Haitang, maybe you can go ahead.
It's very technical.
So let me carefully check this question.
So can I copy this question outside?
Yeah.
Or cannot?
Okay.
So everyone, just so you note that actually for our speakers,
when they look at the chat box, it's not the most user-friendly.
Sometimes it moves really, really quickly.
So they need some time to read.
Okay, so give them some time.
And then so they're able to digest the question
and then come back with an insightful response.
So thanks, everybody.
Okay.
So for the first question from Rohan,
I understand that maybe the logs of the agents are very long.
And you asked, can you just submit the final agent loop?
Is that correct?
Okay.
I think it depends, because actually,
the logs of this tech channel,
it's not very large compared to our daily work,
the logs in our daily work.
It depends on whether you would like to submit all the logs
or just the final logs.
And currently, we also process very long.
For example, like that in our daily work,
we may have maybe several gigabytes of logs
that we need to process.
And we also use AI or AI agent
to extract the most important information from the log.
So actually, it depends on whether you would like to display
all the logs or just the final logs.
It depends on you.
And for the second question, final model training data.
So the second question is a very good question.
Actually, according to my experience,
even though you may think that if finally you train your model
on the training data and validation data,
you may get a better performance on the validation.
But according to my experience,
if you try to tune your model on your validation data,
the performance of your model will dramatically drop
in the test set, in the outside test set.
So actually, I really recommend to you
to only train on the training set
and tune the model on the validation set.
Actually, I do not encourage you to finally train
on both the training set and the validation set.
If you finally train on both,
I guess the performance of your model will dramatically drop.
And on the other side,
if you're trying to train on both sets,
it is not a good strategy in your daily work.
And the second, the log random file.
It's overlapping.
It's using it.
So Rohan asked again that can you just...
Actually, I mean that you can submit all your logs
and you...
But actually, you will write clearly in your report
which is the best one.
Actually, we will score according to your best model,
but we also carefully consider how you use the agent
to improve the performance of the model.
So this is why we also want to see the logs
in the intermediate stage.
So is it clear?
So for the third question of Rohan,
maybe you can just write an email to Phyllis
and I will try to reply this email by today.
Okay, so let's get to the next question.
So the next question is,
is it true an autonomous model
with mediocre scores
will do better than a very accurate one
with significant more intervention?
Actually, you can imagine that
if you are really familiar with the recommender system
and like if you have a very experienced
machine engineer to help you
with some very good intervention,
the model will get...
I think the model will get much better,
but actually, in this challenge,
we consider both.
First, your model needs to have some improvement
on the baseline.
And also, we also consider
how you use the autonomous agent
to improve your model with consider both.
So maybe you ask that...
So it depends on you.
You may use more human intervention
to get a higher score,
but in this stage, in this method,
your method will not be very autonomous.
And we say, oh, maybe you are very familiar
with the recommender system,
but actually, we also want you
to do more autonomous agent to improve.
Actually, we consider both.
It depends on how you balance the two objectives.
Actually, I do not think
restarting a crashed process is...
Let me check.
Actually, you can use...
For example, you may create a session in Cloud Code
to do the autonomous loop.
You can also create another session
to help you to restart this crashed process.
I think that only the change of behavior
is counted as human intervention.
Because...
Let me think.
We know that the AI agent
will suddenly crash
because of some network issues.
For this kind of restarting,
you can use another session to help you to restart,
or maybe you want to do the manual start.
We do not consider this as a manual intervention.
We only consider the manual intervention
if you change the agent's behavior.
For the first, I think it's very simple
to implement with another session of agent.
Is it clear?
I think it's the last response.
Okay.
Is there a time limit for the video?
What should be the format?
You can directly ask Philips.
Yeah, correct.
The video is just recommended to be three minutes long.
Free API keys.
Actually, if you want to use more tokens,
you can try to register for the buy-downs agent
tool called Trail.
It provides a seven-day free trial for the pro version.
Hai-Chan, I think we ran out of time.
Thanks everybody for submitting your questions.
You can also write in your questions,
but our engineers also will not be able to answer every single one of them.
Definitely ask questions where they're relevant to the problem
and also to help you to be successful.
But at the same time, read the problem statement carefully
and understand what you need to also be able to be bold
and innovate and also dare to try with different approaches.
We are wishing you all the best.
Maybe just some final reminders.
You're all set for this hackathon.
Just a quick reminder that track two actually has a very detailed
judging criteria that you want to refer to in the information document
for the problem statement.
Hai-Chan has really, really taken time to explain
exactly what each of the criteria actually would encompass.
So please be careful and read through
and to understand all these criteria.
So this is an overview of the hackathon journey and timeline.
Do note that you must register if you haven't yet.
You must register on our dev post and our form
to be an eligible participant in this hackathon.
And you must also submit your projects by 1st September, 12 o'clock p.m.
So 1st September noon is the deadline for our registration
and also our project submission via dev post.
So do note that lead entries will not be considered
and we're very strict on the deadline.
So all the best with that.
So today we also have a few back-to-back webinars
and next up at 3 o'clock we have track three,
implement a GPU kernel for a transformer layer with Hao Da Li.
So do join us later.
So these are the useful resources.
I'm sure everyone is already very familiar,
but we have our dev post page, our registration form,
information document with all the details of all the problem statements,
as well as our telegram channel.
And our telegram channel, you can actually stay updated,
you know, with all the real-time updates
that we might have and announcements as well.
So we thank you so much for being part of TechGem2026
and we hope to see you in the next webinar.
Thank you.
