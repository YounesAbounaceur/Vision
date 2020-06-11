from flask import Flask, request, render_template, make_response
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import smtplib


app = Flask(__name__)

@app.route('/')
def render_home():
    data = pd.read_csv('HardDataSet.csv',low_memory=False,encoding='latin-1')
    list_skills = data['Skills']
    list_jobs = data['Title']
    list_skills_of_given_job = get_top_skills_of_specific_job(list_skills, list_jobs)
    final_list = recommand_skills([list_skills_of_given_job],20)
    return render_template('index.html', list_top_skills = final_list)
    

@app.route('/job-listings.html', methods=['POST'])
def render_search_job():
    user_skills = request.form.getlist('competence_job[]')

    data = pd.read_csv('HardDataSet.csv',low_memory=False,encoding='latin-1')
    list_companies = data['Company_Name']
    list_jobs = data['Title']
    list_skills = data['Skills']
    list_Links = data['Link']
    list_result_jobs = get_perfect_job(list_companies,list_jobs,list_skills, list_Links, user_skills)
    resp = make_response(render_template('job-listings.html', list_jobs = list_result_jobs, submission_successful = 'true', user_skills = user_skills))
    
    cookie_skills = ""
    for skill in user_skills:
        cookie_skills = cookie_skills+ "^"+ skill
    resp.set_cookie('user_skills', cookie_skills)
   
    return resp
    #return render_template('job-listings.html', list_jobs = list_result_jobs, submission_successful = 'true')


@app.route('/certificats-listings.html', methods=['POST'])
def render_search_certificats():
    user_job = [request.form['user_job']]

    data = pd.read_csv('HardDataSet.csv',low_memory=False,encoding='latin-1')
    list_jobs = data['Title']
    list_skills = data['Skills']

    general = get_similar_jobs(user_job,get_unique_jobs(list_jobs))
    general = sorted(general, reverse = True)
    given_job = general[0][1][0]
    list_skills_of_given_job = get_skills_of_specific_job(given_job,list_skills,list_jobs)
    result_list = recommand_skills(list_skills_of_given_job,20)
    
    return render_template('certificats-listings.html', list_result_skills = result_list, nombre_skills = len(result_list), submission_successful = 'true')

@app.route('/contact.html', methods=['POST'])
def send_email():
    rec_email = "benimmarmary@gmail.com"
    sender_email = "webvisionapp@gmail.com"
    password = "Vision_123"
    
    message = 'Subject: {}\n\n Email : {} \n Message :{}'.format(request.form['subject'], request.form['email'] , request.form['message'])
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, rec_email, message)
    return render_template('contact.html', submission_successful = 'true')

@app.route('/index.html')
def index():
    data = pd.read_csv('HardDataSet.csv',low_memory=False,encoding='latin-1')
    list_skills = data['Skills']
    list_jobs = data['Title']
    list_skills_of_given_job = get_top_skills_of_specific_job(list_skills, list_jobs)
    final_list = recommand_skills([list_skills_of_given_job],20)
    return render_template('index.html', list_top_skills = final_list)

@app.route('/job-listings.html')
def jobListing():
    if 'user_skills' in request.cookies:
        cookie_skills = request.cookies.get('user_skills')
        user_skills= []
        for skill in cookie_skills.split("^") :
            skill = skill.strip().capitalize()
            if skill != "":
                user_skills.append(skill)    
    else:
        user_skills = [""]
    return render_template('job-listings.html', user_skills = user_skills)
    
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/certificats-listings.html')
def certificats_listings():
    return render_template('certificats-listings.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')


def get_unique_jobs(list_jobs):
    unique_jobs = []
    for job in list_jobs:
        if(job not in unique_jobs):
            unique_jobs.append(job)
    return unique_jobs

def get_score_job(job,vector_user_job,vectorizer):
    vector_unique_job = vectorizer.transform(job)
    list_similarity = linear_kernel(vector_user_job,vector_unique_job).flatten()
    score_similarity = sum(list_similarity)/len(list_similarity)
    return score_similarity,job

def get_similar_jobs(user_job,unique_jobs):

    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    vectorizer.fit(unique_jobs)
    list_general = []
    vector_user_job = vectorizer.transform(user_job)
    for job in unique_jobs:
        job = [job]
        list_general.append(get_score_job(job,vector_user_job,vectorizer))
    return list_general

def get_skills_of_specific_job(given_job,list_skills,list_jobs):
    list_skills_of_given_job = []
    for i in range(0,len(list_jobs)):
        if list_jobs[i] == given_job:
            list_cleaned = list_skills[i].replace(";"," ")
            list_skills_of_given_job.append(list_cleaned)
    return list_skills_of_given_job

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
  
def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx] 
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def recommand_skills(my_list,topN):
    #generate tf-idf for the given document
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    cv = vectorizer.fit(my_list)
    feature_names = cv.get_feature_names()
    tf_idf_vector = []
   #apply tf-idf vectorizer on all the skills
    for i in range(len(my_list)):
        text = [my_list[i]]
        tf_idf_vector = vectorizer.transform(text)
        #gather all the vectors in one vector
        if i == 0 :
            final_element = tf_idf_vector
        else :
            final_element = final_element + tf_idf_vector
    #sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(final_element.tocoo())
    #extract the top n keywords
    recommended_skills = extract_topn_from_vector(feature_names,sorted_items,topN)

    recommended_skills = list(map(lambda x:x.replace("_", " ").title(),recommended_skills ))
    return recommended_skills


def get_top_skills_of_specific_job(list_skills, list_jobs):
    list_skills_of_given_job = []
    key_element = ""
    for i in range(0,len(list_jobs)):
        list_cleaned = list_skills[i].replace(";"," ")
        list_skills_of_given_job.append(list_cleaned)
        key_element = key_element + list_cleaned
    return key_element
  
def get_similarity_scrore_for_each_post(company_name,job_title,user_vector,vectorizer,job_required_skils, job_link):
    job_required_skils = [job_required_skils]
    job_vector = vectorizer.transform(job_required_skils)
    from sklearn.metrics.pairwise import linear_kernel
    list_similarity = linear_kernel(job_vector,user_vector).flatten()
    score=sum(list_similarity)/len(list_similarity)
    return score,company_name,job_title, job_link

def get_perfect_job(list_companies,list_jobs,list_skills, list_Links, skills_of_user):
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    vectorizer.fit(list_skills)
    user_vector = vectorizer.transform(skills_of_user)
    list_post_score =[]
    
    for i in range(0,len(list_jobs)):
        S = get_similarity_scrore_for_each_post(list_companies[i],list_jobs[i],user_vector,vectorizer,list_skills[i], list_Links[i])
        list_post_score.append(S)
        
    list_post_score = sorted(list_post_score, reverse = True)
    i=0
    result_list = []
    for element in list_post_score:
        if i < 20:
            i = i+1
            link = str(element[3])
            if link =="nan" :
                link = ""
            result_list.append([element[1].replace("_"," ").title(), element[2], link])
        else: break
        
        
    return result_list


if __name__ == "__main__":
    app.run()
    app.config['TESTING'] = True