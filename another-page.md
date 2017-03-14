---
layout: default
use_math: true
---



This is a tutorial on getting started with logistic regression.  Some of the questions to answer:
- What is logistic regression?
- What are some applications?
- What are the assumptions?
- Special data transformations?
- What are the limitations?

## A simple example
Before we get into the details of logistic regression, let's start with an example.  

Imagine that you work for the marketing department at a tech startup, and you have a wealth of data from your Google Analytics *link here* account.  The majority of the company's marketing is done via the site's blog.  Your goal is to improve the conversion rate of blog viewers into email subscribers.  After brainstorming what factors you think might influence people to subscribe, you come up with the following two:
1. article length (number of words)
2. number of inbound links (links to the blog post from other sites)

After coming up with your ideas, you tidy up your data into a table.  You have thousands of results, but you just want to take a look at the first few rows to get a feel for the data.

article length|number of inbound links|subscribed
:---: | :---: | :---:
523 | 4 | no
2146|58|yes
1148|23|no
1894|49|yes


$$
   |\psi_1\rangle = a|0\rangle + b|1\rangle
$$
[back](./)
