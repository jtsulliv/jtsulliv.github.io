---
title:  "Automated Job Discovery and Evaluation: Web Scraping Indeed.com"
date:   2016-09-23
tags: [web scraping]

header:
  image: "indeed_scraping/job_discovery_header.jpg"
  caption: "Photo credit: [**Novartis**](https://www.nibr.com/files/job-discovery-0)"

excerpt: "Job Discovery, Web Scraping, Indeed.com"
---

I'm getting ready to start looking for a new job (my term at the Federal Reserve Board lasts 2 years), so I've been trying to reflect on what's important to me in a job. I could write thousands of words about how I want to challenge myself intellectually, make a difference in people's lives, work with smart and passionate people, and many other things -- but I won't.

Finding a job like that isn't easy. It requires seeking out and applying to specific jobs that I've identified as a strong fit based on my skillset and desire for self-actualization. Though that is how I'm going to look for jobs, I'm not even **close** to a good enough writer to make such a post remotely readable.

So, instead, this post will be about the opposite side of the job search: finding potentially interesting jobs from the abyss that I might not have otherwise seen. I'll focus on [Indeed.com](http://www.indeed.com/), a major job aggregator that is one of the top stops in many people's job search.

# Overview

Okay, so there are thousands of jobs posted to Indeed.com every day. It's ridiculous to think I could go through them all and evaluate whether I'm a good fit for every single job. It would take forever.


So I need to create some kind of automated pipeline that will:

1. Sift through the recent job results on Indeed
2. Evaluate each of the jobs
3. Identify the ones that are relevant to me
4. Email me a list of the recent jobs that are relevant to me every day

Then I'll stick this in my crontab and just check my email every day for new data science jobs. Since I want to be able to tweak many of the parameters (including the job search terms), I'm going to build this pipeline through a series of stand-alone functions.

# Evaluating a Single Job

While it's not easy to perfectly evaluate any data science job, it's fairly trivial to see if it's remotely relevant to my skillset. For example, if I were 100% focused on getting a job requiring Python programming, I probably wouldn't apply to jobs whose descriptions didn't have the word Python in them. By following this (simple) heuristic, I can write a function to do the evaluation for me. The function will need to search the HTML of a job page and return whether or not I'm a good match. For me, the function should check if Python, R, and SQL are in the job description and return the number of times those terms appear.


```python
import re
import bs4
import time
import requests
import smtplib

def evaluate_job(job_url):
    try:
        job_html = requests.request('GET', job_url, timeout = 10)
    except:
        return 0
    
    job_soup = bs4.BeautifulSoup(job_html.content, 'lxml')
    soup_body = job_soup('body')[0]
    
    python_count = soup_body.text.count('Python') + soup_body.text.count('python')
    sql_count = soup_body.text.count('SQL') + soup_body.text.count('sql')
    r_count = len(re.findall('R[\,\.]', soup_body.text)) # this one's not perfect, but I blame R's name
    skill_count = python_count + sql_count + r_count
    print 'R count: {0}, Python count: {1}, SQL count: {2}'.format(r_count, python_count, sql_count)
    
    return skill_count
```

Let's evaluate a sample job page. How about the [Senior Associate Data Scientist position at Illinois Technology Association](http://www.jobs.net/jobs/ita/en-us/job/United-States/Senior-Associate-Data-Scientist/J3H08S62X9XZZTYDXJP/), the first non-sponsored result on my Indeed.com results page.


```python
evaluate_job('http://www.jobs.net/jobs/ita/en-us/job/United-States/Senior-Associate-Data-Scientist/J3H08S62X9XZZTYDXJP/')
```

    R count: 1, Python count: 1, SQL count: 2
    4



Nice! I got hits for R, Python, and SQL. Looks like this function is working (and that I might be a good fit for this job).

# Getting Job Data from a single Indeed.com Page

Now that we have a function to evaluate a job, we need a function to get the list of jobs to evaluate from a page on Indeed.com. I don't just want the URL (though that's all I need to evaluate the job), since I'm also interested in attributes like the title, company name, and date the job was posted.

Using `requests` and `beautifulsoup` again, I can extract the non-sponsored jobs on every page. These jobs have a special attribute in their `<div>` statement, `data-tn-component="organicJob"`, which let's me get them pretty easily. Then, I just extract the relevant attributes for each job (as a dictionary) and return a list of the job dictionaries.


```python
def extract_job_data_from_indeed(base_url):
    response = requests.get(base_url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    
    tags = soup.find_all('div', {'data-tn-component' : "organicJob"})
    companies_list = [x.span.text for x in tags]
    attrs_list = [x.h2.a.attrs for x in tags]
    dates = [x.find_all('span', {'class':'date'}) for x in tags]
    
    # update attributes dictionaries with company name and date posted
    [attrs_list[i].update({'company': companies_list[i].strip()}) for i, x in enumerate(attrs_list)]
    [attrs_list[i].update({'date posted': dates[i][0].text.strip()}) for i, x in enumerate(attrs_list)]
    return attrs_list
```

Let's look at a sample job attribute dictionary (the first one on the page)


```python
extract_job_data_from_indeed('http://www.indeed.com/jobs?q=data+scientist&l=New+York%2C+NY&sort=date')[0]
```

    {'class': ['turnstileLink'],
     'company': u'Illinois Technology Association',
     'data-tn-element': 'jobTitle',
     'date posted': u'Just posted',
     'href': '/rc/clk?jk=f858e2d65923d3fc&fccid=3de83442785d5fca',
     'itemprop': 'title',
     'onclick': 'setRefineByCookie([]); return rclk(this,jobmap[0],true,0);',
     'onmousedown': 'return rclk(this,jobmap[0],0);',
     'rel': ['nofollow'],
     'target': '_blank',
     'title': 'Senior Associate, Data Scientist'}



Awesome! There's some extraneous information in here, but I'm not that worried about memory usage so it's not an issue. I've got the company name, date posted, hyperlink to the job, and the job title. I'm good to go.

# Finding and Evaluating all the New Jobs

With these functions in hand, I'm almost ready to find and evaluate new jobs. So far I've only evaluated jobs based on the programming languages. This makes sense, since I'm interested in data science, but I might also be interested in jobs at specific companies (regardless of the job description). I'll make a list of these companies and I'll use it as another way to evaluate a job. If a company name matches one in the list, I'll treat it as a relevant job. As an example, I'll just pick the big five US tech companies.


```python
extra_interest_companies = ['apple', 'microsoft', 'google', 'facebook', 'amazon']
```

Okay, I can extract job information from a page on Indeed, and then I can evaluate the individual jobs from their URLs. Now I need to loop through Indeed.com's newest results for a search query and apply my functions to every page. I'll define a function `find_new_jobs` to do this.


```python
def find_new_jobs(days_ago_limit = 1, starting_page = 0, pages_limit = 20, old_jobs_limit = 5,
                  location = 'New York, NY', query = 'data scientist'):
    
    query_formatted = re.sub(' ', '+', query)
    location_formatted = re.sub(' ', '+', location)
    indeed_url = 'http://www.indeed.com/jobs?q={0}&l={1}&sort=date&start='.format(query_formatted, location_formatted)
    old_jobs_counter = 0
    new_jobs_list = []
    
    for i in xrange(starting_page, starting_page + pages_limit):
        if old_jobs_counter >= old_jobs_limit:
            break
        
        print 'URL: {0}'.format(indeed_url + str(i*10)), '\n'

        # extract job data from Indeed page
        attrs_list = extract_job_data_from_indeed(indeed_url + str(i*10))
        
        # loop through each job, breaking out if we're past the old jobs limit
        for j in xrange(0, len(attrs_list)): 
            if old_jobs_counter >= old_jobs_limit:
                break

            href = attrs_list[j]['href']
            title = attrs_list[j]['title']
            company = attrs_list[j]['company']
            date_posted = attrs_list[j]['date posted']
            
            # if posting date is beyond the limit, add to the counter and skip
            try:
                if int(date_posted[0]) >= days_ago_limit:
                    print 'Adding to old_jobs_counter.'
                    old_jobs_counter+= 1
                    continue
            except:
                pass

            print '{0}, {1}, {2}'.format(repr(company), repr(title), repr(date_posted))

            # evaluate the job
            evaluation = evaluate_job('http://indeed.com' + href)
            
            if evaluation >= 1 or company.lower() in extra_interest_companies:
                new_jobs_list.append('{0}, {1}, {2}'.format(company, title, 'http://indeed.com' + href))
                
            print '\n'
            time.sleep(15)
            
    new_jobs_string = '\n\n'.join(new_jobs_list)
    return new_jobs_string
```

There's some extra stuff in there to be respectful of Indeed's servers and to make sure I'm only evaluating recent jobs, but essentially it's just a loop around the two functions I created earlier.

# Emailing Myself the New Jobs

The one weird thing about the function I defined just now is the output. I had a list of jobs, where each element in the list is a tuple in the format of `(company name, job title, url)`, but I returned it as a string (separated by two new lines). By returning the jobs in this format, it will look better when I email it to myself.

Using the `smtplib` library makes emailing really easy. I'll wrap the code into a function so I can send emails easily in different scenarios.


```python
def send_gmail(from_addr = '****', to_addr = '****',
               location = 'New York, NY',
               subject = 'Daily Data Science Jobs Update Scraped from Indeed', text = None):
    
    message = 'Subject: {0}\n\nJobs in: {1}\n\n{2}'.format(subject, location, text)

    # login information
    username = '****'
    password = '****'
    
    # send the message
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(from_addr, to_addr, message)
    server.quit()
    print 'Email sent.'
```

# Putting it all Together

So, we're all ready.

Since I may want to import the functions I created into another program sometime, I want to make sure the code only runs when I execute the code as a stand-alone program, as opposed to if I import it into another program. The `if __name__ == "__main__"` let's me do just that.

Time for a test run.


```python
def main():
    print 'Scraping Indeed now.'

    start_page = 0
    page_limit = 2
    location = 'New York, NY'
    data_scientist_jobs = find_new_jobs(query = 'data scientist', starting_page = start_page,
                                        location = location, pages_limit = page_limit, days_ago_limit = 1, old_jobs_limit = 5)
    send_gmail(text = data_scientist_jobs, location = location)
```


```python
if __name__ == "__main__":
    main()
```

    Scraping Indeed now.
    URL: http://www.indeed.com/jobs?q=data+scientist&l=New+York,+NY&sort=date&start=0 
    
    u'Illinois Technology Association', 'Data Scientist', u'Just posted'
    R count: 1, Python count: 1, SQL count: 2

    [...]

    u'Weill Cornell Medical College', 'Research Technician I', u'Just posted'
    R count: 4, Python count: 0, SQL count: 0
    
    
    Email sent.


Easy. Here's a snapshot of the email I sent myself.

![png](/images/indeed_scraping/indeed_email_pic.png?raw=True)

# Conclusion

Now I can just set up a cron-job and I'll get an email every day with newly posted data science jobs I might be interested in. If I want to get emails with jobs in other cities or from other Indeed.com search terms, I can just add another `find_new_jobs` and `send_gmail` couplet to the `main` function. If anyone was wondering why I was writing all of this code in functions (instead of as a stand-alone program), that's the main reason right there.

While this is all pretty useful, it's important to remember that finding jobs this way is externally rather than internally oriented. It's looking at the set of all (many) possibilities and finding ones that might be relevant, as opposed to looking to see if there are any openings at companies and organizations that I find exciting. As a result, this is just a supplement to my actual job search.

It's finding jobs that I might be a good technical fit for -- not jobs that might be a good fit for me. It's a small distinction, but a massive difference.
