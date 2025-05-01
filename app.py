
import pandas as pd
import bcrypt
import json
import os
import time
import datetime
from streamlit_cookies_manager import CookieManager
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
from torch.nn.functional import cosine_similarity
import streamlit as st

st.set_page_config(page_title="AI Job Portal", page_icon="📄")

# ---- Paths ----
DATA_DIR = "D:\\PycharmProjects\\pythonProject4\\DS_PROJECT\\AI - Driven Skill Based\\data"
RESUME_DIR = "D:\\PycharmProjects\\pythonProject4\\DS_PROJECT\\AI - Driven Skill Based\\resumes"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESUME_DIR, exist_ok=True)

USER_DB = os.path.join(DATA_DIR, "users.json")
JOB_CSV = os.path.join(DATA_DIR, "jobs.csv")
RESUME_CSV = os.path.join(DATA_DIR, "resumes_data.csv")

# ---- Init Cookie Manager ----
cookies = CookieManager()
if not cookies.ready():
    st.stop()

# ---- Helpers ----
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(USER_DB, "w") as file:
        json.dump(users, file, indent=4)

def authenticate(username, password):
    users = load_users()
    if username in users and check_password(password, users[username]["password"]):
        return users[username]["role"]
    return None

def register_user(username, password, role):
    users = load_users()
    users[username] = {"password": hash_password(password), "role": role}
    save_users(users)

# ---- Auth Page ----
def auth_page():
    st.title("Login / Signup")
    choice = st.radio("Choose:", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        role = st.selectbox("Select Role", ["Job Seeker", "Employer"])
        if st.button("Register"):
            if username in load_users():
                st.error("Username already exists!")
            else:
                register_user(username, password, role)
                st.success("Account created! Please login.")
                time.sleep(1)
                st.rerun()

    if choice == "Login":
        if st.button("Login"):
            role = authenticate(username, password)
            if role:
                cookies['username'] = username
                cookies['role'] = role
                cookies.save()
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials!")

# ---- Employer Dashboard ----
def employer_dashboard():
    st.title("Manage Your Jobs")

    username = cookies.get("username")

    if os.path.exists(JOB_CSV):
        df = pd.read_csv(JOB_CSV)
    else:
        df = pd.DataFrame(columns=["Employer", "Title", "Company", "Location", "Experience", "Skills", "Salary", "Type"])

    st.subheader("Post a New Job")
    with st.form("post_job"):
        title = st.text_input("Job Title")
        company = st.text_input("Company Name")
        location = st.text_input("Location")
        experience = st.number_input("Experience Required (Years)", min_value=0)
        skills = st.text_area("Skills (comma-separated)")
        salary = st.text_input("Salary")
        job_type = st.selectbox("Job Type", ["Full-Time", "Part-Time", "Internship", "Contract"])
        submitted = st.form_submit_button("Post Job")

    if submitted:
        new_job = pd.DataFrame([{
            "Employer": username,
            "Title": title,
            "Company": company,
            "Location": location,
            "Experience": experience,
            "Skills": skills,
            "Salary": salary,
            "Type": job_type
        }])
        df = pd.concat([df, new_job], ignore_index=True)
        df.to_csv(JOB_CSV, index=False)
        st.success("Job posted successfully!")

    st.subheader("Your Posted Jobs")
    employer_jobs = df[df['Employer'] == username]
    if employer_jobs.empty:
        st.info("You haven't posted any jobs.")
    else:
        for idx, job in employer_jobs.iterrows():
            with st.expander(f"{job['Title']} at {job['Company']}"):
                edited_job = st.text_input("Job Title", job['Title'])
                edited_location = st.text_input("Location", job['Location'])
                edited_experience = st.number_input("Experience Required", min_value=0, value=job['Experience'])
                edited_skills = st.text_area("Skills", job['Skills'])
                edited_salary = st.text_input("Salary", job['Salary'])
                edited_type = st.selectbox("Job Type", ["Full-Time", "Part-Time", "Internship", "Contract"], index=["Full-Time", "Part-Time", "Internship", "Contract"].index(job['Type']))
                if st.button(f"Update {job['Title']}"):
                    df.at[idx, 'Title'] = edited_job
                    df.at[idx, 'Location'] = edited_location
                    df.at[idx, 'Experience'] = edited_experience
                    df.at[idx, 'Skills'] = edited_skills
                    df.at[idx, 'Salary'] = edited_salary
                    df.at[idx, 'Type'] = edited_type
                    df.to_csv(JOB_CSV, index=False)
                    st.success("Job updated successfully!")
                    st.rerun()

                if st.button(f"Delete {job['Title']}"):
                    df = df.drop(idx)
                    df.to_csv(JOB_CSV, index=False)
                    st.warning("Job deleted!")
                    st.rerun()


def upload_resume():
    with st.form("resume_form"):
        name = st.text_input("Full Name",placeholder="ABC XYZ",value=cookies.get("username"))
        email = st.text_input("Email",placeholder="abc@gmail.com")
        phone = st.text_input("Phone Number",placeholder="+91 0123456789")
        linkedin = st.text_input("LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/username")
        github = st.text_input("GitHub Profile URL", placeholder="https://github.com/username")

        # Key Resume Sections
        education = st.text_area("Education", placeholder="Add degrees, institutions, and year of passing")
        experience = st.text_area("Experience", placeholder="Mention roles, companies, duration, and responsibilities")
        certifications = st.text_area("Certifications", placeholder="List your certifications with year")
        skills = st.text_area("Skills (comma separated)", placeholder="e.g., Python, Machine Learning, SQL, Excel")
        projects = st.text_area("Projects", placeholder="Briefly mention your key projects")
        achievements = st.text_area("Achievements", placeholder="Mention awards, recognitions, publications, etc.")

        # File Upload (Resume File)
        resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

        submit = st.form_submit_button("Submit Resume")

    if submit:
        if all([name, email, phone, skills, education, experience, resume_file]):
            # Save resume file to a folder
            if resume_file:
                file_path = os.path.join(RESUME_DIR, resume_file.name)
                with open(file_path, "wb") as f:
                    f.write(resume_file.getbuffer())

            # Prepare Data for CSV
            new_data = pd.DataFrame([[
                name, name, email, phone, linkedin, github,
                education, experience, certifications, 
                skills, projects, achievements, 
                file_path
            ]], columns=[
                "Username","Name", "Email", "Phone", "LinkedIn", "GitHub", 
                "Education", "Experience", "Certifications", 
                "Skills", "Projects", "Achievements", 
                "Resume_File"
            ])

            # Append to CSV
            df = pd.read_csv(RESUME_CSV)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(RESUME_CSV, index=False)

            st.success("Resume and all details saved successfully!")

        else:
            st.error("Please fill in all mandatory fields (Name, Email, Phone, Skills, Education, Experience, Resume File).")

def resume_dashboard():
    st.title("Manage Your Resume")

    username = cookies.get("username")

    if os.path.exists(RESUME_CSV):
        df = pd.read_csv(RESUME_CSV)
    else:
        df = pd.DataFrame(columns=["Username", "Name", "Email", "Phone", "LinkedIn", "GitHub", "Education", "Experience", "Certifications", "Skills", "Projects", "Achievements", "Resume_File"])

    user_resume = df[df['Username'] == username]

    if user_resume.empty:
        st.warning("No resume found! Please upload.")
        upload_resume()
    else:
        idx = user_resume.index[0]
        st.subheader("Edit Your Resume")
        df.at[idx, 'Name'] = st.text_input("Name", user_resume['Name'].values[0])
        df.at[idx, 'Email'] = st.text_input("Email", user_resume['Email'].values[0])
        df.at[idx, 'Phone'] = st.text_input("Phone", user_resume['Phone'].values[0])
        df.at[idx, 'Skills'] = st.text_area("Skills", user_resume['Skills'].values[0])
        df.at[idx, 'Education'] = st.text_area("Education", user_resume['Education'].values[0])
        df.at[idx, 'Experience'] = st.text_area("Experience", user_resume['Experience'].values[0])
        resume_file_path = user_resume['Resume_File'].values[0]
        # st.text_input("Resume",resume_file_path[77:])
        if os.path.exists(resume_file_path):
            with open(resume_file_path, "rb") as f:
                st.download_button("Download Current Resume", f, file_name=f"{resume_file_path[77:]}")
        else:
            st.warning("No resume file found in system! Please upload a new file.")

        # Option to replace with a new resume file
        new_resume_file = st.file_uploader("Upload New Resume (Optional - Replaces Current)", type=["pdf","docx"])

        if st.button("Update Resume"):
            if new_resume_file:
                # Replace the existing resume file
                resume_path = os.path.join(RESUME_DIR, f"{resume_file_path[77:]}")
                with open(resume_path, "wb") as f:
                    f.write(new_resume_file.read())
                df.at[idx, 'Resume_File'] = resume_path
                st.success("New resume uploaded and replaced successfully!")

            # Always save the other updated fields
            df.to_csv(RESUME_CSV, index=False)
            st.success("Resume updated successfully!")

def employ_job_dashboard():
    st.title("Employer Dashboard")

    username = cookies.get("username", None)
    role = cookies.get("role", None)

    if not username or role != "Employer":
        st.warning("Access Denied: Employers only.")
        return

    # Load job applications
    if not os.path.exists("job_applications.csv"):
        st.warning("No applications found.")
        return

    applications_df = pd.read_csv("job_applications.csv")

    # Load job postings
    if not os.path.exists(JOB_CSV):
        st.warning("No jobs posted yet.")
        return

    jobs_df = pd.read_csv(JOB_CSV)

    # Get jobs posted by this employer
    employer_jobs = jobs_df[jobs_df["Employer"] == username]["Title"].tolist()

    if not employer_jobs:
        st.warning("You haven't posted any jobs yet.")
        return

    # Filter applications for the employer's jobs
    employer_applications = applications_df[applications_df["Job Title"].isin(employer_jobs)]

    if employer_applications.empty:
        st.warning("No applicants for your jobs yet.")
        return

    # Display applicants
    for _, app in employer_applications.iterrows():
        st.markdown(f"### {app['Job Title']} - Applicant: {app['Username']}")
        st.write(f"**Applied On:** {app['Applied Date']}")
        st.write(f"**Current Status:** `{app['Status']}`")

        # Accept & Reject Buttons
        if app["Status"] == "Pending":
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Accept {app['Username']}", key=f"accept_{app['Username']}_{app['Job Title']}"):
                    applications_df.loc[(applications_df["Username"] == app["Username"]) & 
                                        (applications_df["Job Title"] == app["Job Title"]), "Status"] = "Accepted"
                    applications_df.to_csv("job_applications.csv", index=False)
                    st.success(f"Accepted {app['Username']} for {app['Job Title']}")

            with col2:
                if st.button(f"Reject {app['Username']}", key=f"reject_{app['Username']}_{app['Job Title']}"):
                    applications_df.loc[(applications_df["Username"] == app["Username"]) & 
                                        (applications_df["Job Title"] == app["Job Title"]), "Status"] = "Rejected"
                    applications_df.to_csv("job_applications.csv", index=False)
                    st.error(f"Rejected {app['Username']} for {app['Job Title']}")

        st.markdown("---")


model = SentenceTransformer('all-MiniLM-L6-v2')  

def job_seeker_dashboard():
    st.title("Job Dashboard")

    # Load user session
    username = cookies.get("username", None)
    role = cookies.get("role", None)  # Get the user role

    if not username:
        st.warning("Please login to access this page.")
        return

    # Load job postings
    if not os.path.exists(JOB_CSV):
        st.warning("No jobs posted yet.")
        return

    jobs_df = pd.read_csv(JOB_CSV)

    if jobs_df.empty:
        st.warning("No jobs found in the database.")
        return
    
    # If the user is an employer, show all jobs
    if role != "Job Seeker":
        st.subheader("All Job Listings")
        for _, job in jobs_df.iterrows():
            st.markdown(f"### {job['Title']} at {job['Company']}")
            st.write(f"**Location:** {job['Location']}")
            st.write(f"**Required Skills:** {job['Skills']}")
            st.write(f"**Experience Required:** {job['Experience']} years")
            st.write(f"**Salary:** ₹{job['Salary']} per month")
            st.write(f"**Job Type:** {job['Type']}")
            st.markdown("---")
        return

    # Load user resume data
    if not os.path.exists(RESUME_CSV):
        st.warning("No resumes found. Please upload your resume.")
        return

    resume_df = pd.read_csv(RESUME_CSV)
    user_data = resume_df[resume_df["Username"] == username]

    if user_data.empty:
        st.warning("Please upload your resume first.")
        return

    # Extract user skills
    user_skills_text = user_data.iloc[0]["Skills"]

    # Compute similarity using embeddings
    user_embedding = model.encode([user_skills_text], convert_to_tensor=True)
    jobs_df["job_text"] = jobs_df["Title"] + " " + jobs_df["Skills"] + " " + jobs_df["Experience"].astype(str)
    job_embeddings = model.encode(jobs_df["job_text"].tolist(), convert_to_tensor=True)
    similarities = cosine_similarity(user_embedding, job_embeddings)

    # Get relevant jobs
    threshold = 0.54  
    k = min(5, len(jobs_df))
    best_matches = torch.topk(similarities, k=k)
    filtered_jobs = [idx for idx in best_matches.indices if similarities[idx] > threshold]
    matched_jobs = jobs_df.iloc[filtered_jobs] if filtered_jobs else pd.DataFrame()

    st.subheader("Recommended Jobs for You")

    if matched_jobs.empty:
        st.info("No suitable jobs found based on your profile.")
    else:
        for _, job in matched_jobs.iterrows():
            st.markdown(f"### {job['Title']} at {job['Company']}")
            st.write(f"**Location:** {job['Location']}")
            st.write(f"**Required Skills:** {job['Skills']}")
            st.write(f"**Experience Required:** {job['Experience']} years")
            st.write(f"**Salary:** ₹{job['Salary']} per month")
            st.write(f"**Job Type:** {job['Type']}")

            # Load previous applications
            if os.path.exists("job_applications.csv"):
                applications_df = pd.read_csv("job_applications.csv")
                user_applied = (applications_df[(applications_df["Username"] == username) & 
                                                (applications_df["Job Title"] == job["Title"])]).empty
            else:
                user_applied = True

            # Apply Now Button
            if user_applied:
                if st.button(f"Apply Now for {job['Title']}", key=job["Title"] + username):
                    application_data = {
                        "Username": username, 
                        "Job Title": job["Title"], 
                        "Company": job["Company"],
                        "Applied Date": datetime.datetime.today().strftime('%Y-%m-%d'), 
                        "Status": "Pending"
                    }
                    pd.DataFrame([application_data]).to_csv("job_applications.csv", mode='a', 
                                                            header=not os.path.exists("job_applications.csv"), 
                                                            index=False)
                    st.success(f"You applied for {job['Title']}!")

            st.markdown("---")


def my_applications():
    st.title("My Applications")

    username = cookies.get("username", None)

    if not username:
        st.warning("Please login to access this page.")
        return

    # Load job applications
    if not os.path.exists("job_applications.csv"):
        st.warning("You have not applied for any jobs yet.")
        return

    applications_df = pd.read_csv("job_applications.csv")
    user_applications = applications_df[applications_df["Username"] == username]

    if user_applications.empty:
        st.warning("You have not applied for any jobs yet.")
        return

    # Display applied jobs
    for _, app in user_applications.iterrows():
        st.markdown(f"### {app['Job Title']} at {app['Company']}")
        st.write(f"**Applied On:** {app['Applied Date']}")
        st.write(f"**Status:** `{app['Status']}`")  # Pending, Accepted, Rejected
        st.markdown("---")


def trending_skills_dashboard():
    """ Display personalized trending skills based on job market demand using K-Means clustering. """
    st.title("🚀 Trending Skills in the Market")

    # Load user session
    username = cookies.get("username", None)
    if not username:
        st.warning("Please login to access this page.")
        return

    # Load user resume data
    if not os.path.exists(RESUME_CSV):
        st.warning("No resumes found. Please upload your resume.")
        return

    resume_df = pd.read_csv(RESUME_CSV)
    user_data = resume_df[resume_df["Username"] == username]

    if user_data.empty:
        st.warning("User resume not found. Please upload your resume.")
        return

    # Load job postings
    if not os.path.exists(JOB_CSV):
        st.warning("No jobs posted yet.")
        return

    jobs_df = pd.read_csv(JOB_CSV)
    
    if jobs_df.empty:
        st.warning("No jobs found in the database.")
        return

    # Extract user skills
    user_skills_text = user_data.iloc[0]["Skills"]
    user_skills = {skill.strip().lower() for skill in user_skills_text.split(",")}

    # Extract job skills
    job_skills = jobs_df["Skills"].dropna().tolist()  # Remove NaN values

    # Convert job skills into numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(job_skills)

    # Apply K-Means clustering to group jobs
    num_clusters = min(5, len(job_skills))  # Adjust for small datasets
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    jobs_df["Cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Find the most relevant cluster for the user
    user_vector = vectorizer.transform([user_skills_text])
    user_cluster = kmeans.predict(user_vector)[0]

    # Filter jobs only from this cluster
    cluster_jobs = jobs_df[jobs_df["Cluster"] == user_cluster]

    # Extract trending skills only from this cluster
    trending_skills = set()
    for skills in cluster_jobs["Skills"]:
        trending_skills.update(skill.strip().lower() for skill in skills.split(","))

    # 🔹 **Find skills the user is missing**
    trending_skills.discard("")  # Remove empty skills
    missing_skills = trending_skills - user_skills

    # Display trending skills (only for relevant jobs)
    st.subheader("Trending Skills in Your Career Path")
    st.write(", ".join(sorted(trending_skills)))

    # Display missing skills
    if missing_skills:
        st.subheader("Recommended Skills to Learn (Just for You)")
        st.info("To improve your chances of getting hired, consider learning these skills:")
        st.write(", ".join(sorted(missing_skills)))
    else:
        st.success("Great job! You already have trending skills in your profile.")

# ---- Project Explanation ----
def project_explanation():
    st.title("Project Explanation")
    st.write("""
This project is designed to help both **Job Seekers** and **Employers** through a smart matching system powered by **Machine Learning algorithms**.

### Project Goals
- Collect candidate resumes via form.
- Store resumes in structured format (CSV).
- Employers can post jobs with required skills.
- The system will match resumes to jobs based on **skill matching** and **cosine similarity**.
- A recommendation system will suggest jobs to candidates based on their submitted profile.

---

### Algorithms Used
- **CountVectorizer** for text processing.
- **Cosine Similarity** for resume-job matching.
- **KNN (K-Nearest Neighbors)** for personalized job recommendations.
- **K-Means Clustering** for grouping jobs into categories.
- **Rule-based Filtering** for basic keyword matching.

---

### System Modules
- Resume Collection  
- Job Posting by Employers  
- Job Matching & Recommendations  
- Job Display for Candidates

---

### Future Scope
- Enhancing recommendations using user behavior
- Adding feedback loop to improve matching quality
""")

# ---- Dashboard ----
def dashboard():
    username = cookies.get("username")
    role = cookies.get("role")

    st.sidebar.title(f"Welcome, {username}")
    st.sidebar.write(f"Role: {role}")

    if st.sidebar.button("Logout"):
        cookies['username'] = ""
        cookies['role'] = ""
        cookies.save()
        st.rerun()

    if role == "Employer":
        page = st.sidebar.radio("Navigation", ["Project Explanation", "Manage Jobs","Job Openings","Manage Application"])
        if page == "Manage Jobs":
            employer_dashboard()
        if page == "Job Openings":
            job_seeker_dashboard()
        if page == "Manage Application":
            employ_job_dashboard()
    else:
        page = st.sidebar.radio("Navigation", ["Project Explanation", "Manage Resume","Job Openings","Trending Skills","My Application"])
        if page == "Manage Resume":
            resume_dashboard()
        if page == "Job Openings":
            job_seeker_dashboard()
        if page == "Trending Skills":
            trending_skills_dashboard()
        if page == "My Application":
            my_applications()
    if page == "Project Explanation":
        project_explanation()

# ---- Main Entry ----
def main():
    if cookies.get("username"):
        dashboard()
    else:
        auth_page()

if __name__ == "__main__":
    main()
