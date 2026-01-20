"""
Career data including roadmaps, job search links, and reference lists
"""

# Education levels (based on your dataset)
EDUCATION_LEVELS = [
    "PhD",
    "Master's",
    "MBA",
    "Bachelor's",
    "Associate's",
    "Diploma",
    "Intermediate",
    "High School"
]

# Common skills from your dataset
SKILLS_LIST = [
    # Programming Languages
    "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "TypeScript", "Swift", "Kotlin",
    "Ruby", "PHP", "Scala", "R", "SQL", "Solidity",
    
    # Web Development
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Express.js", "HTML", "CSS",
    "REST APIs", "GraphQL", "WebSockets", "Redux", "Next.js",
    
    # Mobile Development
    "React Native", "Flutter", "Android", "iOS", "Jetpack Compose", "SwiftUI", "Firebase",
    
    # Data Science & ML
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas",
    "NumPy", "Computer Vision", "NLP", "Neural Networks", "Reinforcement Learning", "Data Mining",
    "Statistical Analysis", "Predictive Modeling", "Feature Engineering",
    
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins", "Terraform",
    "Ansible", "Chef", "DevOps", "CloudFormation", "Serverless", "Microservices",
    
    # Databases
    "MongoDB", "PostgreSQL", "MySQL", "Redis", "Cassandra", "Oracle", "SQL Server",
    "DynamoDB", "Neo4j", "Database Design", "Data Modeling",
    
    # Big Data
    "Hadoop", "Spark", "Kafka", "Data Pipelines", "ETL", "Data Warehousing", "Hive", "Flink",
    
    # Security
    "Cybersecurity", "Penetration Testing", "Ethical Hacking", "Network Security",
    "Security Auditing", "Threat Analysis", "Encryption", "SIEM", "Firewall Configuration",
    
    # Design
    "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator", "UI Design", "UX Design",
    "Wireframing", "Prototyping", "Visual Design", "Interaction Design", "Design Systems",
    "User Research",
    
    # Blockchain
    "Ethereum", "Smart Contracts", "Web3", "DeFi", "Truffle", "Hardhat",
    
    # Game Development
    "Unity", "Unreal Engine", "3D Modeling", "Game Design", "C++ Game Development",
    "Shader Programming", "Physics Simulation", "Animation",
    
    # Testing
    "Unit Testing", "Integration Testing", "E2E Testing", "Test Automation", "Selenium",
    "JUnit", "Jest", "Cypress", "Performance Testing", "QA",
    
    # Data Analysis
    "Data Analysis", "Data Visualization", "Tableau", "Power BI", "Excel", "Statistics",
    "A/B Testing", "Business Intelligence", "Metrics", "Analytics",
    
    # Other Technical Skills
    "Git", "Linux", "Networking", "System Design", "Algorithms", "Data Structures",
    "API Development", "Agile", "Scrum", "Project Management", "Technical Writing",
    "Problem Solving", "Code Review", "Architecture Design", "Solution Architecture"
]

# Common interests
INTERESTS_LIST = [
    "Artificial Intelligence", "Machine Learning", "Software Development", "Web Development",
    "Mobile Development", "Data Science", "Cloud Computing", "Cybersecurity", "Blockchain",
    "Game Development", "UI/UX Design", "DevOps", "Big Data", "IoT", "Robotics",
    "Computer Vision", "Natural Language Processing", "Automation", "Testing",
    "Network Security", "System Administration", "Database Management", "Product Development",
    "Innovation", "Technology", "Problem Solving", "Research", "Analytics",
    "Infrastructure", "Mathematics", "Physics", "Business Strategy", "Entrepreneurship",
    "User Experience", "Human-Computer Interaction", "Psychology", "Market Analysis",
    "Product Management", "Team Leadership", "Cryptocurrency", "Decentralization",
    "Gaming", "Graphics Programming", "3D Design", "Animation", "Virtual Reality",
    "Augmented Reality", "API Development", "Microservices", "Distributed Systems",
    "Performance Optimization", "Code Quality", "Open Source", "Community Building"
]

# Career roadmaps
CAREER_ROADMAPS = {
    "AI Research Scientist": [
        {
            "title": "Master Fundamental Mathematics & Statistics",
            "description": "Study linear algebra, calculus, probability theory, and statistical inference. These form the mathematical foundation of AI research.",
            "duration": "6-12 months"
        },
        {
            "title": "Learn ML/DL Frameworks & Theory",
            "description": "Master TensorFlow/PyTorch, study neural networks, optimization algorithms, and various ML paradigms. Complete online courses and implement papers from scratch.",
            "duration": "12-18 months"
        },
        {
            "title": "Pursue Advanced Degree (MS/PhD)",
            "description": "Enroll in a graduate program focusing on AI/ML. Conduct original research, publish papers, and collaborate with experienced researchers.",
            "duration": "2-5 years"
        },
        {
            "title": "Contribute to Research & Publications",
            "description": "Publish papers at top conferences (NeurIPS, ICML, CVPR), contribute to open-source AI projects, and build a research portfolio.",
            "duration": "Ongoing"
        },
        {
            "title": "Specialize & Lead Research Projects",
            "description": "Develop expertise in a specific AI domain (NLP, computer vision, reinforcement learning) and lead research initiatives.",
            "duration": "3-5 years"
        }
    ],
    
    "Data Scientist": [
        {
            "title": "Learn Programming & Statistics",
            "description": "Master Python/R, SQL, and statistical analysis. Learn data manipulation with Pandas and visualization with Matplotlib/Seaborn.",
            "duration": "3-6 months"
        },
        {
            "title": "Study Machine Learning Algorithms",
            "description": "Learn supervised/unsupervised learning, regression, classification, clustering, and ensemble methods. Complete ML projects.",
            "duration": "6-9 months"
        },
        {
            "title": "Build Portfolio with Real Projects",
            "description": "Work on Kaggle competitions, create end-to-end ML projects, and document them on GitHub. Focus on business impact.",
            "duration": "6-12 months"
        },
        {
            "title": "Master Data Engineering & Big Data",
            "description": "Learn Spark, data pipelines, ETL processes, and cloud platforms (AWS/GCP/Azure) for handling large-scale data.",
            "duration": "6-9 months"
        },
        {
            "title": "Specialize & Lead Data Initiatives",
            "description": "Develop domain expertise, mentor junior data scientists, and drive data strategy for business decisions.",
            "duration": "2-4 years"
        }
    ],
    
    "Software Engineer": [
        {
            "title": "Master Programming Fundamentals",
            "description": "Learn data structures, algorithms, and at least two programming languages (e.g., Python, Java, JavaScript). Practice on LeetCode/HackerRank.",
            "duration": "6-12 months"
        },
        {
            "title": "Learn Web Development / Backend Systems",
            "description": "Study web frameworks, databases, APIs, and version control (Git). Build full-stack applications and deploy them.",
            "duration": "6-12 months"
        },
        {
            "title": "Build Project Portfolio & Contribute to Open Source",
            "description": "Create 3-5 significant projects showcasing different skills. Contribute to open-source projects to gain experience.",
            "duration": "6-12 months"
        },
        {
            "title": "Study System Design & Architecture",
            "description": "Learn about scalability, microservices, distributed systems, and design patterns. Practice system design interviews.",
            "duration": "6-9 months"
        },
        {
            "title": "Specialize & Advance to Senior Roles",
            "description": "Develop expertise in specific domains (backend, frontend, distributed systems) and take on architectural responsibilities.",
            "duration": "3-5 years"
        }
    ],
    
    "Data Engineer": [
        {
            "title": "Learn Programming & Databases",
            "description": "Master Python/Java, SQL, and database design. Understand relational and NoSQL databases (PostgreSQL, MongoDB).",
            "duration": "4-6 months"
        },
        {
            "title": "Study Data Warehousing & ETL",
            "description": "Learn data modeling, ETL processes, and data warehousing concepts. Work with tools like Apache Airflow.",
            "duration": "6-9 months"
        },
        {
            "title": "Master Big Data Technologies",
            "description": "Learn Hadoop, Spark, Kafka, and distributed computing. Build data pipelines for large-scale data processing.",
            "duration": "9-12 months"
        },
        {
            "title": "Cloud Data Platforms & DevOps",
            "description": "Master cloud data services (AWS Redshift, Google BigQuery, Azure Synapse) and implement CI/CD for data pipelines.",
            "duration": "6-9 months"
        },
        {
            "title": "Architect Data Infrastructure",
            "description": "Design scalable data architectures, optimize data pipelines, and establish data governance practices.",
            "duration": "2-4 years"
        }
    ],
    
    "Machine Learning Engineer": [
        {
            "title": "Programming & Software Engineering",
            "description": "Master Python, software engineering practices, Git, testing, and deployment. Learn to write production-quality code.",
            "duration": "6-9 months"
        },
        {
            "title": "Study ML Algorithms & Deep Learning",
            "description": "Learn ML theory, implement algorithms from scratch, and master frameworks like TensorFlow/PyTorch.",
            "duration": "9-12 months"
        },
        {
            "title": "MLOps & Model Deployment",
            "description": "Learn Docker, Kubernetes, model serving (TensorFlow Serving, MLflow), and CI/CD for ML systems.",
            "duration": "6-9 months"
        },
        {
            "title": "Build Production ML Systems",
            "description": "Create end-to-end ML pipelines, implement monitoring, A/B testing, and model retraining strategies.",
            "duration": "12-18 months"
        },
        {
            "title": "Scale & Optimize ML Infrastructure",
            "description": "Design distributed training systems, optimize inference performance, and architect ML platforms.",
            "duration": "2-4 years"
        }
    ],
    
    "DevOps Engineer": [
        {
            "title": "Learn Linux & Scripting",
            "description": "Master Linux administration, shell scripting (Bash), and Python for automation tasks.",
            "duration": "3-6 months"
        },
        {
            "title": "CI/CD & Version Control",
            "description": "Learn Git, GitHub Actions, Jenkins, GitLab CI. Implement automated build and deployment pipelines.",
            "duration": "4-6 months"
        },
        {
            "title": "Containerization & Orchestration",
            "description": "Master Docker for containerization and Kubernetes for container orchestration and management.",
            "duration": "6-9 months"
        },
        {
            "title": "Infrastructure as Code",
            "description": "Learn Terraform, Ansible, or CloudFormation. Automate infrastructure provisioning and configuration.",
            "duration": "6-9 months"
        },
        {
            "title": "Cloud & SRE Practices",
            "description": "Master cloud platforms (AWS/Azure/GCP), implement monitoring, logging, and SRE principles for reliability.",
            "duration": "1-2 years"
        }
    ],
    
    "Full Stack Developer": [
        {
            "title": "Frontend Development Fundamentals",
            "description": "Master HTML, CSS, JavaScript, and a modern framework (React, Vue, or Angular). Build responsive UIs.",
            "duration": "4-6 months"
        },
        {
            "title": "Backend Development & APIs",
            "description": "Learn Node.js/Python/Java for backend, build RESTful APIs, and understand authentication/authorization.",
            "duration": "4-6 months"
        },
        {
            "title": "Databases & State Management",
            "description": "Master SQL and NoSQL databases, learn state management (Redux, Context API), and caching strategies.",
            "duration": "3-6 months"
        },
        {
            "title": "Full Stack Projects & Deployment",
            "description": "Build complete applications from frontend to backend, deploy to cloud platforms, and implement CI/CD.",
            "duration": "6-12 months"
        },
        {
            "title": "System Design & Architecture",
            "description": "Learn scalable architecture patterns, microservices, and lead full-stack development initiatives.",
            "duration": "2-3 years"
        }
    ],
    
    "Cybersecurity Analyst": [
        {
            "title": "Networking & OS Fundamentals",
            "description": "Study TCP/IP, network protocols, Linux/Windows administration, and command-line tools.",
            "duration": "3-6 months"
        },
        {
            "title": "Security Concepts & Tools",
            "description": "Learn cryptography, threat modeling, and security tools (Wireshark, Nmap, Metasploit). Get CompTIA Security+ certified.",
            "duration": "6-9 months"
        },
        {
            "title": "Ethical Hacking & Penetration Testing",
            "description": "Study vulnerability assessment, penetration testing methodologies. Pursue CEH or OSCP certification.",
            "duration": "9-12 months"
        },
        {
            "title": "Security Operations & Incident Response",
            "description": "Learn SIEM tools, log analysis, incident response procedures, and security monitoring techniques.",
            "duration": "6-12 months"
        },
        {
            "title": "Advanced Security Specializations",
            "description": "Specialize in areas like malware analysis, threat intelligence, or cloud security. Pursue CISSP certification.",
            "duration": "2-4 years"
        }
    ],
    
    "Cloud Architect": [
        {
            "title": "Master Cloud Fundamentals",
            "description": "Learn core services of AWS/Azure/GCP, networking, storage, and compute. Get cloud practitioner certification.",
            "duration": "3-6 months"
        },
        {
            "title": "Cloud Solutions & Services",
            "description": "Study serverless, containers, databases, and security in the cloud. Build multi-tier cloud applications.",
            "duration": "6-9 months"
        },
        {
            "title": "Get Cloud Certifications",
            "description": "Pursue AWS Solutions Architect, Azure Solutions Architect, or GCP Professional Architect certifications.",
            "duration": "6-12 months"
        },
        {
            "title": "Design Cloud Architectures",
            "description": "Learn architectural patterns, disaster recovery, cost optimization, and well-architected frameworks.",
            "duration": "12-18 months"
        },
        {
            "title": "Lead Cloud Transformation",
            "description": "Architect enterprise-scale cloud solutions, lead migration projects, and establish cloud governance.",
            "duration": "2-4 years"
        }
    ],
    
    "UI/UX Designer": [
        {
            "title": "Design Fundamentals",
            "description": "Study design principles, color theory, typography, and composition. Learn Figma, Sketch, or Adobe XD.",
            "duration": "3-6 months"
        },
        {
            "title": "User Research & Psychology",
            "description": "Learn user research methods, usability testing, and cognitive psychology. Understand user behavior and needs.",
            "duration": "4-6 months"
        },
        {
            "title": "Interaction & Visual Design",
            "description": "Master wireframing, prototyping, and visual design. Create design systems and maintain consistency.",
            "duration": "6-9 months"
        },
        {
            "title": "Build Design Portfolio",
            "description": "Complete 3-5 significant UX projects showcasing your design process from research to final product.",
            "duration": "6-12 months"
        },
        {
            "title": "Lead Design Strategy",
            "description": "Develop expertise in design thinking, lead design sprints, and establish UX practices in organizations.",
            "duration": "2-3 years"
        }
    ],
    
    "Mobile Developer": [
        {
            "title": "Choose Platform & Language",
            "description": "Master Swift/iOS or Kotlin/Android, or learn cross-platform frameworks (React Native/Flutter).",
            "duration": "4-6 months"
        },
        {
            "title": "Mobile UI & App Architecture",
            "description": "Learn mobile UI patterns, navigation, state management, and architectural patterns (MVVM, MVI, Clean Architecture).",
            "duration": "6-9 months"
        },
        {
            "title": "Backend Integration & APIs",
            "description": "Master REST/GraphQL APIs, local databases, authentication, and offline-first architecture.",
            "duration": "4-6 months"
        },
        {
            "title": "Build & Publish Apps",
            "description": "Create complete apps, test thoroughly, and publish to App Store/Play Store. Gather user feedback.",
            "duration": "6-12 months"
        },
        {
            "title": "Performance & Advanced Features",
            "description": "Optimize performance, implement push notifications, payments, analytics, and advanced platform features.",
            "duration": "1-2 years"
        }
    ],
    
    "Product Manager": [
        {
            "title": "Business & Technical Fundamentals",
            "description": "Learn product management frameworks, basic technical concepts, data analysis, and business strategy.",
            "duration": "3-6 months"
        },
        {
            "title": "User Research & Market Analysis",
            "description": "Master user research methods, competitive analysis, market sizing, and customer development.",
            "duration": "4-6 months"
        },
        {
            "title": "Product Strategy & Roadmapping",
            "description": "Learn to define product vision, create roadmaps, prioritize features, and work with stakeholders.",
            "duration": "6-9 months"
        },
        {
            "title": "Launch Products & Iterate",
            "description": "Lead product launches, define metrics, run A/B tests, and iterate based on data and user feedback.",
            "duration": "1-2 years"
        },
        {
            "title": "Senior PM & Strategy",
            "description": "Own product strategy, manage multiple products, mentor junior PMs, and drive business outcomes.",
            "duration": "3-5 years"
        }
    ],
    
    "Blockchain Developer": [
        {
            "title": "Learn Blockchain Fundamentals",
            "description": "Study blockchain technology, consensus mechanisms, cryptography, and distributed systems.",
            "duration": "3-4 months"
        },
        {
            "title": "Master Solidity & Smart Contracts",
            "description": "Learn Solidity programming, develop smart contracts, and understand Ethereum ecosystem.",
            "duration": "6-9 months"
        },
        {
            "title": "Development Tools & Testing",
            "description": "Master Truffle, Hardhat, Web3.js, and smart contract testing frameworks. Learn security best practices.",
            "duration": "4-6 months"
        },
        {
            "title": "Build DApps & DeFi Projects",
            "description": "Create decentralized applications, implement DeFi protocols, and integrate with blockchain networks.",
            "duration": "9-12 months"
        },
        {
            "title": "Advance to Protocol Development",
            "description": "Contribute to blockchain protocols, design tokenomics, and architect blockchain solutions.",
            "duration": "2-3 years"
        }
    ],
    
    "Game Developer": [
        {
            "title": "Programming & Math Foundations",
            "description": "Master C++ or C#, learn linear algebra, trigonometry, and physics for game development.",
            "duration": "6-9 months"
        },
        {
            "title": "Game Engine Proficiency",
            "description": "Learn Unity or Unreal Engine, understand game loops, rendering pipelines, and asset management.",
            "duration": "6-12 months"
        },
        {
            "title": "Game Design & Mechanics",
            "description": "Study game design principles, level design, player experience, and game balancing.",
            "duration": "6-9 months"
        },
        {
            "title": "Complete Game Projects",
            "description": "Build 2-3 complete games from concept to release, publish on Steam/mobile stores, and gather feedback.",
            "duration": "12-18 months"
        },
        {
            "title": "Specialize & Lead Projects",
            "description": "Develop expertise in specific areas (graphics, AI, multiplayer) and lead game development teams.",
            "duration": "2-4 years"
        }
    ],
    
    "Backend Developer": [
        {
            "title": "Master Server-Side Language",
            "description": "Learn Node.js, Python (Django/Flask), Java (Spring), or Go. Understand async programming and concurrency.",
            "duration": "4-6 months"
        },
        {
            "title": "Databases & APIs",
            "description": "Master SQL/NoSQL databases, design RESTful APIs, implement authentication, and handle data persistence.",
            "duration": "6-9 months"
        },
        {
            "title": "System Design & Architecture",
            "description": "Learn microservices, caching strategies, message queues, and distributed systems patterns.",
            "duration": "6-12 months"
        },
        {
            "title": "DevOps & Cloud Deployment",
            "description": "Master Docker, CI/CD, cloud platforms, and implement monitoring and logging for backend services.",
            "duration": "6-9 months"
        },
        {
            "title": "Scale & Optimize Systems",
            "description": "Design high-performance systems, optimize database queries, and architect scalable backend infrastructure.",
            "duration": "2-3 years"
        }
    ],
    
    "Frontend Developer": [
        {
            "title": "HTML, CSS & JavaScript Mastery",
            "description": "Master modern JavaScript (ES6+), CSS Grid/Flexbox, responsive design, and web accessibility.",
            "duration": "4-6 months"
        },
        {
            "title": "Modern Framework Expertise",
            "description": "Learn React, Vue, or Angular in depth. Master state management, component architecture, and hooks/composition API.",
            "duration": "6-9 months"
        },
        {
            "title": "Advanced Frontend Concepts",
            "description": "Study performance optimization, build tools (Webpack, Vite), testing (Jest, Testing Library), and TypeScript.",
            "duration": "6-9 months"
        },
        {
            "title": "Build Production Applications",
            "description": "Create responsive, accessible web applications with great UX. Learn SEO, PWAs, and deployment strategies.",
            "duration": "9-12 months"
        },
        {
            "title": "Frontend Architecture & Leadership",
            "description": "Design component libraries, establish coding standards, and lead frontend development initiatives.",
            "duration": "2-3 years"
        }
    ],
    
    "Network Engineer": [
        {
            "title": "Networking Fundamentals",
            "description": "Study OSI model, TCP/IP, routing, switching, and network protocols. Get CompTIA Network+ certified.",
            "duration": "3-6 months"
        },
        {
            "title": "Cisco/Network Certifications",
            "description": "Pursue CCNA certification, learn Cisco IOS, configure routers and switches, and understand VLANs and routing protocols.",
            "duration": "6-12 months"
        },
        {
            "title": "Network Security & Monitoring",
            "description": "Learn firewalls, VPNs, network security principles, and monitoring tools (Wireshark, SolarWinds).",
            "duration": "6-9 months"
        },
        {
            "title": "Advanced Networking & Automation",
            "description": "Study SDN, network automation with Python, and advanced routing protocols. Consider CCNP certification.",
            "duration": "12-18 months"
        },
        {
            "title": "Network Architecture & Design",
            "description": "Design enterprise networks, implement redundancy and high availability, and lead network infrastructure projects.",
            "duration": "2-4 years"
        }
    ],
    
    "QA Engineer": [
        {
            "title": "Software Testing Fundamentals",
            "description": "Learn testing methodologies, test case design, bug reporting, and QA processes. Understand SDLC.",
            "duration": "2-4 months"
        },
        {
            "title": "Test Automation Basics",
            "description": "Learn programming (Python/JavaScript), version control, and basic automation frameworks (Selenium, Cypress).",
            "duration": "4-6 months"
        },
        {
            "title": "Advanced Test Automation",
            "description": "Master test automation frameworks, CI/CD integration, API testing (Postman, RestAssured), and performance testing.",
            "duration": "6-9 months"
        },
        {
            "title": "Specialized Testing Skills",
            "description": "Develop expertise in specific areas: mobile testing, security testing, or performance testing (JMeter, LoadRunner).",
            "duration": "9-12 months"
        },
        {
            "title": "QA Leadership & Strategy",
            "description": "Establish QA processes, mentor testers, and implement quality engineering practices across teams.",
            "duration": "2-3 years"
        }
    ],
}

def get_career_roadmap(career_name):
    """Get roadmap for a specific career"""
    return CAREER_ROADMAPS.get(career_name, [])

def get_job_search_links(career_name):
    """Generate job search links for a specific career"""
    # URL encode the career name for search queries
    career_query = career_name.replace(" ", "+")
    
    return {
        "LinkedIn": f"https://www.linkedin.com/jobs/search/?keywords={career_query}",
        "Indeed": f"https://www.indeed.com/jobs?q={career_query}",
        "Glassdoor": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={career_query}",
        "Monster": f"https://www.monster.com/jobs/search/?q={career_query}",
        "SimplyHired": f"https://www.simplyhired.com/search?q={career_query}",
        "ZipRecruiter": f"https://www.ziprecruiter.com/Jobs/{career_query}"
    }
