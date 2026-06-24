"""
Script to generate a synthetic resume dataset for training.
Generates 2400+ resumes across 5 job categories.
"""

import pandas as pd
import random
import re

random.seed(42)

# ── Templates per category ─────────────────────────────────────────────────

TEMPLATES = {
    "Data Scientist": {
        "skills": [
            "Python", "R", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn", "SQL",
            "Statistics", "Data Analysis", "NLP", "Computer Vision", "A/B Testing",
            "Feature Engineering", "XGBoost", "Keras", "Jupyter Notebook", "Tableau",
            "Spark", "Hadoop", "Data Visualization", "Regression", "Classification",
            "Clustering", "Neural Networks", "Model Deployment", "MLflow", "Azure ML",
        ],
        "roles": [
            "Data Scientist", "ML Engineer", "Research Scientist", "AI Engineer",
            "Data Science Intern", "Junior Data Scientist", "Senior Data Scientist",
        ],
        "edu": ["B.Sc Statistics", "M.Sc Data Science", "B.Tech CS", "Ph.D Machine Learning", "B.Sc Mathematics"],
        "phrases": [
            "built predictive models to forecast customer churn",
            "developed NLP pipeline for sentiment analysis",
            "implemented recommendation system using collaborative filtering",
            "performed exploratory data analysis on large datasets",
            "trained deep learning models for image classification",
            "optimized model performance using hyperparameter tuning",
            "deployed ML models to production using Flask and Docker",
            "conducted A/B testing and statistical hypothesis testing",
            "created dashboards for business intelligence reporting",
            "analyzed user behavior data to improve product features",
        ]
    },
    "Web Developer": {
        "skills": [
            "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js",
            "Express.js", "Bootstrap", "Tailwind CSS", "TypeScript", "REST API",
            "GraphQL", "MongoDB", "MySQL", "PHP", "WordPress", "Figma", "Git",
            "Webpack", "jQuery", "SASS", "Redux", "Next.js", "Responsive Design",
            "Web Accessibility", "SEO", "JSON", "Firebase", "Vercel", "Netlify",
        ],
        "roles": [
            "Web Developer", "Frontend Developer", "Full Stack Developer",
            "UI Developer", "Web Design Intern", "Junior Web Developer",
        ],
        "edu": ["B.Sc Computer Science", "B.Tech IT", "Diploma Web Development", "B.E Software Engineering"],
        "phrases": [
            "developed responsive web applications using React and Node.js",
            "built e-commerce platform with payment gateway integration",
            "designed and implemented RESTful APIs for mobile applications",
            "optimized website performance improving load time by 40%",
            "created interactive UI components using JavaScript and CSS",
            "maintained and updated client websites using WordPress CMS",
            "integrated third-party APIs for maps, payments, and social login",
            "implemented user authentication and authorization using JWT",
            "built progressive web apps with offline functionality",
            "collaborated with designers to convert mockups to code",
        ]
    },
    "Software Engineer": {
        "skills": [
            "Java", "C++", "Python", "C#", "Go", "Rust", "Data Structures", "Algorithms",
            "OOP", "Design Patterns", "System Design", "Microservices", "Docker",
            "Kubernetes", "CI/CD", "Git", "Linux", "AWS", "Azure", "GCP",
            "Unit Testing", "Agile", "Scrum", "SQL", "NoSQL", "Redis", "Kafka",
            "Spring Boot", "REST API", "gRPC", "Jenkins", "Terraform", "Bash",
        ],
        "roles": [
            "Software Engineer", "Backend Developer", "Systems Engineer",
            "DevOps Engineer", "SDE Intern", "Junior Software Engineer", "SDE-I",
        ],
        "edu": ["B.Tech Computer Science", "B.E Software Engineering", "M.Tech CS", "B.Sc CS"],
        "phrases": [
            "designed and developed scalable backend services handling 10M+ requests",
            "implemented microservices architecture reducing system latency by 30%",
            "wrote clean, maintainable code following SOLID principles",
            "optimized database queries improving performance by 50%",
            "built CI/CD pipelines using Jenkins and GitHub Actions",
            "developed RESTful APIs consumed by web and mobile clients",
            "resolved critical production bugs and improved system reliability",
            "participated in code reviews and mentored junior developers",
            "containerized applications using Docker and Kubernetes",
            "designed system architecture for high-availability distributed systems",
        ]
    },
    "HR": {
        "skills": [
            "Recruitment", "Talent Acquisition", "Onboarding", "Employee Relations",
            "Performance Management", "HR Policies", "HRIS", "Payroll", "Labor Law",
            "Training & Development", "Conflict Resolution", "Compensation & Benefits",
            "Applicant Tracking System", "Workday", "SAP HR", "Communication",
            "Leadership", "MS Office", "Excel", "Interviewing", "Job Posting",
            "Background Verification", "Employee Engagement", "Exit Interviews",
        ],
        "roles": [
            "HR Executive", "HR Manager", "Talent Acquisition Specialist",
            "HR Coordinator", "Recruiter", "HR Intern", "HR Business Partner",
        ],
        "edu": ["MBA HR", "B.Sc Psychology", "BBA", "MBA Business Administration", "M.Sc Organizational Behavior"],
        "phrases": [
            "managed end-to-end recruitment for technical and non-technical roles",
            "conducted structured interviews and assessed candidate fit",
            "developed onboarding programs improving new hire retention by 25%",
            "administered performance appraisal cycles for 500+ employees",
            "resolved employee grievances and maintained positive work culture",
            "coordinated training programs and leadership development initiatives",
            "maintained HRIS records and ensured data accuracy",
            "partnered with business leaders on workforce planning",
            "implemented employee engagement surveys and action plans",
            "ensured compliance with labor laws and company HR policies",
        ]
    },
    "Marketing": {
        "skills": [
            "Digital Marketing", "SEO", "SEM", "Google Ads", "Facebook Ads",
            "Content Marketing", "Social Media", "Email Marketing", "HubSpot",
            "Salesforce", "Google Analytics", "Market Research", "Brand Management",
            "Copywriting", "CRM", "Campaign Management", "A/B Testing", "Lead Generation",
            "Product Marketing", "Influencer Marketing", "Marketing Automation",
            "Canva", "Adobe Photoshop", "Video Marketing", "KPI Analysis", "Mailchimp",
        ],
        "roles": [
            "Marketing Executive", "Digital Marketing Specialist", "Content Strategist",
            "Brand Manager", "Marketing Intern", "Growth Hacker", "SEO Specialist",
        ],
        "edu": ["MBA Marketing", "BBA Marketing", "B.Sc Mass Communication", "B.A Journalism", "M.Sc Marketing"],
        "phrases": [
            "managed social media accounts growing followers by 150% in 6 months",
            "created and executed digital marketing campaigns achieving 3x ROI",
            "conducted keyword research and implemented SEO strategies",
            "developed content calendar and managed blog with 50K+ monthly readers",
            "ran Google and Facebook ad campaigns with monthly budget of $50K",
            "analyzed marketing data using Google Analytics to optimize campaigns",
            "collaborated with design team to produce marketing collateral",
            "implemented email marketing campaigns with 35% open rate",
            "conducted competitor analysis and market research for product launches",
            "generated 500+ qualified leads per month through inbound marketing",
        ]
    }
}

NAMES = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Gupta", "Rohit Singh",
    "Ananya Verma", "Vikram Mehta", "Pooja Joshi", "Suresh Nair", "Divya Rao",
    "Arjun Kapoor", "Kavya Reddy", "Manish Agarwal", "Neha Pandey", "Rajesh Iyer",
    "Sonia Malhotra", "Deepak Tiwari", "Ritu Sharma", "Aakash Desai", "Meera Pillai",
    "John Smith", "Emily Johnson", "Michael Brown", "Sarah Davis", "James Wilson",
    "Jessica Martinez", "David Anderson", "Jennifer Taylor", "Christopher Lee", "Amanda White",
]

COMPANIES = [
    "Tech Solutions Pvt Ltd", "Infosys", "TCS", "Wipro", "HCL Technologies",
    "Accenture", "Cognizant", "IBM India", "Google India", "Microsoft India",
    "Amazon India", "Flipkart", "Swiggy", "Zomato", "Paytm", "Ola", "Byju's",
    "Startup Hub", "Digital Ventures", "DataTech Corp", "CloudSystems Inc",
]

def generate_resume(category, idx):
    t = TEMPLATES[category]
    name = random.choice(NAMES)
    num_skills = random.randint(8, 16)
    skills = random.sample(t["skills"], num_skills)
    role = random.choice(t["roles"])
    edu = random.choice(t["edu"])
    company = random.choice(COMPANIES)
    exp_years = random.randint(0, 8)
    phrases = random.sample(t["phrases"], random.randint(3, 6))
    gpa = round(random.uniform(6.5, 9.8), 1)

    resume = f"""
{name}
Email: {name.lower().replace(' ', '.')}@email.com | Phone: +91-9{random.randint(100000000, 999999999)}

OBJECTIVE
Motivated {role} with {exp_years} years of experience seeking opportunities in {category}.

EDUCATION
{edu} | {random.randint(2015, 2024)} | GPA: {gpa}/10

EXPERIENCE
{role} | {company} | {random.randint(2018, 2024)} - Present
{' '.join([f'• {p.capitalize()}.' for p in phrases])}

SKILLS
{', '.join(skills)}

PROJECTS
Project {idx % 5 + 1}: Built a {random.choice(['web', 'mobile', 'data', 'automation', 'AI'])} application
using {', '.join(random.sample(skills, min(3, len(skills))))}.
Achieved {random.randint(10, 90)}% improvement in {random.choice(['efficiency', 'accuracy', 'speed', 'performance'])}.

CERTIFICATIONS
{random.choice(['AWS Certified', 'Google Cloud', 'Microsoft Azure', 'Coursera Certificate', 'Udemy Course'])} - {category}
    """.strip()

    return resume

def generate_dataset(n_per_class=500):
    categories = list(TEMPLATES.keys())
    data = []
    for cat in categories:
        for i in range(n_per_class):
            resume = generate_resume(cat, i)
            data.append({"resume": resume, "category": cat})

    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv("resumes.csv", index=False)
    print(f"Generated {len(df)} resumes across {len(categories)} categories.")
    print(df["category"].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset(500)
