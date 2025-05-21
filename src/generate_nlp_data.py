import numpy as np
import pandas as pd
import random
import string
import os
import datetime
import json
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

# ==================== SYNTHETIC DATA GENERATORS ====================

import random
import os
from pathlib import Path
import datetime
import pandas as pd
import re

def generate_synthetic_emails(n_emails=30, output_folder='custom_emails', date_range=('2024-03-01', '2025-04-15'), 
                             domains=None, department_dist=None, save_files=True, seed=123):
    """
    Generate synthetic email data and optionally save to files.
    
    Parameters:
    -----------
    n_emails : int
        Number of emails to generate
    output_folder : str
        Folder to save email files
    date_range : tuple
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    domains : list
        List of email domains (default: common domains)
    department_dist : dict
        Dictionary of department weights for distribution (default: uniform)
    save_files : bool
        Whether to save emails as .eml files
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with email content and metadata
    """
    # Set random seed
    random.seed(seed)
    
    # Default domains if not provided
    domains = domains or ['example.com', 'company.org', 'biz.net']
    
    # Default departments and distribution
    departments = ['IT', 'HR', 'Sales', 'Marketing', 'Finance']
    department_dist = department_dist or {dept: 1/len(departments) for dept in departments}
    
    # Normalize department distribution
    total_weight = sum(department_dist.values())
    department_dist = {k: v/total_weight for k, v in department_dist.items()}
    
    # Define templates
    service_templates = [
        "Salesforce",
        "AWS",
        "Google Cloud",
        "Microsoft Azure",
        "Shopify",
        "Zendesk",
        "HubSpot",
        "Stripe",
        "GitHub",
        "Slack"
    ]
    
    subject_templates = {
        'IT': [
            'Issue with {service} for {client}',
            'Update on {project} for Q{quarter} {year}',
            'New {service} {feature} implementation'
        ],
        'HR': [
            'Training on {topic} for {department}',
            'Feedback on {event}',
            'New {feature} for {department}'
        ],
        'Sales': [
            'New {service} opportunity with {client}',
            'Follow-up on {event} with {client}',
            'Q{quarter} {year} {topic} proposal'
        ],
        'Marketing': [
            '{service} campaign for {client}',
            'Review of {topic} strategy',
            'Launch of {project}'
        ],
        'Finance': [
            'Billing issue with {service} for {client}',
            'Q{quarter} {year} financial report',
            'Update on {feature} process'
        ]
    }
    
    issue_templates = [
        'Problem with {module} access',
        'Error in {module} processing',
        'Issue reported in {module}'
    ]
    
    feature_templates = [
        'New {module} {process}',
        'Updated {module} {process} system',
        'Enhanced {module} {process} feature'
    ]
    
    project_templates = [
        'Project v{version}',
        'Release v{version}',
        '{version} upgrade'
    ]
    
    # Simple email body templates
    body_templates = [
        "Dear {recipient},\n\nWe have an update regarding {subject}.\nPlease review and let us know your feedback.\n\nBest regards,\n{sender}",
        "Hi {recipient},\n\nThe {subject} requires your attention.\nCould you please confirm by {month} {day}?\n\nThanks,\n{sender}",
        "Hello {recipient},\n\nRegarding {subject}, we need to discuss next steps.\nAre you available for a meeting this week?\n\nRegards,\n{sender}"
    ]
    
    # Create output folder
    output_folder = Path(output_folder)
    if save_files:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate sender and recipient names
    first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']
    
    email_info = []
    start_date = datetime.datetime.strptime(date_range[0], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(date_range[1], '%Y-%m-%d')
    date_diff = (end_date - start_date).days
    
    for i in range(n_emails):
        # Select department based on distribution
        random_dept = random.choices(departments, weights=[department_dist[d] for d in departments])[0]
        
        # Select sender and recipient
        sender = f"{random.choice(first_names)} {random.choice(last_names)}"
        recipient = f"{random.choice(first_names)} {random.choice(last_names)}"
        sender_email = f"{sender.lower().replace(' ', '.')}@{random.choice(domains)}"
        recipient_email = f"{recipient.lower().replace(' ', '.')}@{random.choice(domains)}"
        
        # Generate random date
        random_days = random.randint(0, date_diff)
        email_date = start_date + datetime.timedelta(days=random_days)
        date_str = email_date.strftime('%a, %d %b %Y %H:%M:%S +0000')
        
        # Select subject template
        subject_template = random.choice(subject_templates[random_dept])
        
        # Create format arguments
        format_args = {
            'number': random.randint(1000, 9999),
            'quarter': random.randint(1, 4),
            'year': random.randint(2024, 2025),
            'month': random.choice(['January', 'February', 'March', 'April', 'May', 'June', 
                                   'July', 'August', 'September', 'October', 'November', 'December']),
            'department': random_dept,
            'client': random.choice(['Acme Inc.', 'Globex Corp', 'Initech', 'Umbrella Corp', 'Stark Industries']),
            'event': random.choice(['Trade Show', 'Conference', 'Product Launch', 'Workshop', 'Webinar']),
            'topic': random.choice(['Digital Transformation', 'Data Security', 'Customer Experience', 
                                   'Market Expansion', 'Product Development', 'Innovation']),
            'issue': random.choice(issue_templates).format(
                module=random.choice(['Login', 'Dashboard', 'Profile', 'Settings', 'Analytics', 'Payment'])
            ),
            'feature': random.choice(feature_templates).format(
                module=random.choice(['User', 'Admin', 'Analytics', 'Reporting', 'Security']),
                process=random.choice(['onboarding', 'approval', 'reporting', 'invoicing', 'review'])
            ),
            'project': random.choice(project_templates).format(
                version=f"{random.randint(1, 5)}.{random.randint(0, 9)}"
            ),
            'service': random.choice(service_templates)
        }
        
        # Format subject, only using placeholders present in the template
        try:
            placeholders = re.findall(r'\{([^}]+)\}', subject_template)
            subject = subject_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
        except KeyError as e:
            print(f"KeyError in subject formatting: {e}. Using fallback subject.")
            subject = f"Update from {random_dept} on {format_args['topic']}"
        
        # Generate email body
        body_template = random.choice(body_templates)
        body = body_template.format(
            recipient=recipient.split()[0],
            subject=subject,
            sender=sender,
            month=format_args['month'],
            day=random.randint(1, 28)
        )
        
        # Create raw email content
        raw_content = f"From: {sender} <{sender_email}>\n"
        raw_content += f"To: {recipient} <{recipient_email}>\n"
        raw_content += f"Subject: {subject}\n"
        raw_content += f"Date: {date_str}\n\n"
        raw_content += body
        
        # Save email to file
        filename = f"email_{i+1:04d}.eml"
        filepath = output_folder / filename
        if save_files:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(raw_content)
                size_kb = filepath.stat().st_size / 1024
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                size_kb = len(raw_content.encode('utf-8')) / 1024
        else:
            size_kb = len(raw_content.encode('utf-8')) / 1024
        
        # Collect metadata
        info = {
            'filename': filename,
            'path': str(filepath),
            'from': sender_email,
            'to': recipient_email,
            'cc': '',
            'subject': subject,
            'date': date_str,
            'content': body,
            'raw_content': raw_content,
            'word_count': len(body.split()),
            'char_count': len(body),
            'line_count': body.count('\n') + 1,
            'size_kb': size_kb,
            'parsed_date': email_date
        }
        email_info.append(info)
    
    return pd.DataFrame(email_info)

import random
import numpy as np
import pandas as pd
import os
from pathlib import Path
import datetime
import re

def generate_synthetic_documents(n_documents: int = 30,
                               output_folder: str = 'data',
                               categories: Optional[List[str]] = None,
                               date_range: Tuple[str, str] = ('2024-01-01', '2025-05-20'),
                               lengths: Optional[Dict[str, Tuple[int, int]]] = None,
                               save_files: bool = True,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic document data for testing
    
    Parameters:
    -----------
    n_documents : int
        Number of synthetic documents to generate
    output_folder : str
        Folder to save document files (if save_files is True)
    categories : Optional[List[str]]
        List of document categories
    date_range : Tuple[str, str]
        Start and end dates for document timestamps
    lengths : Optional[Dict[str, Tuple[int, int]]]
        Dictionary mapping categories to (min, max) paragraph counts
    save_files : bool
        Whether to save documents as individual files
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with generated document metadata
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Default values
    if categories is None:
        categories = ['report', 'policy', 'proposal', 'analysis', 'memo']
    
    if lengths is None:
        lengths = {
            'report': (10, 20),
            'policy': (8, 15),
            'proposal': (6, 12),
            'analysis': (8, 15),
            'memo': (3, 7)
        }
    
    # Document title templates
    title_templates = {
        'report': [
            "Q{quarter} {year} {department} Report",
            "Annual {department} Report {year}",
            "{project} Project Status Report",
            "{department} Performance Report",
            "Market Research Report: {topic}",
            "Financial Analysis: {quarter} {year}",
            "Technical Report: {project}",
            "User Research Findings: {topic}",
            "Compliance Report: {topic}",
            "Risk Assessment Report: {department}"
        ],
        'policy': [
            "Corporate {topic} Policy",
            "{department} Operational Guidelines",
            "Employee {topic} Policy",
            "Data {topic} Policy",
            "Acceptable Use Policy: {topic}",
            "Information Security Standards",
            "Remote Work Policy {year}",
            "Environmental Sustainability Policy",
            "Ethics and Compliance Guidelines",
            "Vendor Management Policy"
        ],
        'proposal': [
            "Project Proposal: {project}",
            "Business Development: {topic}",
            "Investment Proposal: {project}",
            "Marketing Campaign Proposal: {topic}",
            "Research Grant Proposal: {topic}",
            "New Product Development: {project}",
            "Strategic Partnership Proposal",
            "Cost Reduction Initiative: {department}",
            "Technology Implementation: {project}",
            "Expansion Proposal: {topic}"
        ],
        'analysis': [
            "Competitive Analysis: {topic}",
            "Market Opportunity Analysis",
            "Financial Scenario Analysis",
            "Risk Analysis: {project}",
            "Process Efficiency Analysis",
            "Customer Segmentation Analysis",
            "Cost-Benefit Analysis: {project}",
            "Performance Metrics Analysis",
            "Trend Analysis: {topic}",
            "SWOT Analysis: {department}"
        ],
        'memo': [
            "Internal Memo: {topic}",
            "Meeting Summary: {topic}",
            "Project Update: {project}",
            "Action Items: {department} Meeting",
            "Policy Update Notification",
            "Organizational Announcement",
            "Procedural Change: {topic}",
            "Executive Briefing: {topic}",
            "Budget Allocation Update",
            "Team Restructuring Memo"
        ]
    }
    
    # Content templates by section and category
    content_templates = {
        'introduction': {
            'report': [
                "This report provides an overview of {topic} for the period of {period}. It summarizes key findings, metrics, and recommendations for stakeholders.",
                "The following report presents a comprehensive analysis of {topic}, covering performance metrics, challenges, and strategic recommendations.",
                "This document outlines the results of our investigation into {topic}, including methodology, key findings, and actionable insights."
            ],
            'policy': [
                "This policy establishes guidelines for {topic} within the organization. It defines responsibilities, procedures, and compliance requirements.",
                "The purpose of this policy is to outline standards and procedures for {topic} to ensure consistency, security, and compliance with applicable regulations.",
                "This document defines the organizational approach to {topic}, including scope, responsibilities, and implementation guidelines."
            ],
            'proposal': [
                "This proposal outlines a plan to implement {topic} with the goal of {objective}. It includes background, approach, timeline, and resource requirements.",
                "The following proposal presents a comprehensive strategy for {topic}, including objectives, methodology, expected outcomes, and resource allocation.",
                "This document proposes a new initiative for {topic} to address {challenge} and achieve {objective}."
            ],
            'analysis': [
                "This analysis examines {topic} to identify patterns, trends, and insights. It utilizes data from {period} to draw conclusions and make recommendations.",
                "The following document presents an in-depth analysis of {topic}, including methodology, key findings, and strategic implications.",
                "This analysis evaluates the current state of {topic}, identifies opportunities for improvement, and presents recommendations based on quantitative and qualitative data."
            ],
            'memo': [
                "This memo provides an update on {topic} and outlines next steps for the team.",
                "The purpose of this communication is to inform stakeholders about recent developments regarding {topic} and provide guidance on immediate actions.",
                "This document summarizes key information about {topic} for the attention of {department} team members."
            ]
        },
        'body': {
            'report': [
                "Our analysis shows {finding} with respect to {metric}. This represents a {change} compared to previous periods and indicates {implication}.",
                "The data collected during {period} demonstrates {finding}, which suggests {implication} for our operations and strategic direction.",
                "Based on our evaluation of {metric}, we have identified {finding} that requires attention and strategic response.",
                "Performance indicators for {topic} reveal {finding}, with particular emphasis on {aspect} that has shown significant {change}.",
                "Our investigation into {topic} uncovered {finding}, which has important implications for {department} operations."
            ],
            'policy': [
                "All employees must comply with {requirement} when engaging in activities related to {topic}. Failure to comply may result in {consequence}.",
                "The organization requires strict adherence to {requirement} to ensure {objective} and maintain compliance with {regulation}.",
                "Procedures for {topic} must include {requirement} to safeguard organizational assets and ensure operational integrity.",
                "Department managers are responsible for ensuring {requirement} is implemented consistently across their teams.",
                "Exceptions to this policy may only be granted by {authority} following a formal review process that documents justification and risk mitigation measures."
            ],
            'proposal': [
                "The proposed solution involves {approach} to address {challenge} and achieve {objective}. This approach offers advantages including {benefit}.",
                "We recommend implementing {approach} with a phased rollout beginning in {timeframe}. This strategy minimizes disruption while maximizing impact.",
                "This initiative will require {resource} investment but is projected to yield {benefit} within {timeframe} of implementation.",
                "The key components of our proposed approach include {component}, which addresses the critical needs identified in our preliminary assessment.",
                "Our solution differentiates from alternatives by {differentiator}, providing superior results in terms of {metric}."
            ],
            'analysis': [
                "The data reveals a significant correlation between {variable1} and {variable2}, suggesting that {implication} for our strategic approach.",
                "Our analysis identified {pattern} in the {dataset}, which indicates {insight} that was previously unrecognized.",
                "Comparing performance across {dimension} shows {finding}, with the highest values observed in {category} and lowest in {category}.",
                "The trend analysis for {metric} demonstrates {pattern} over {period}, which contradicts previous assumptions about {topic}.",
                "When segmented by {dimension}, the data shows {finding} that requires a targeted approach for different {category} groups."
            ],
            'memo': [
                "Please note that {topic} will require {action} by {deadline}. This is necessary to ensure {objective}.",
                "The recent development regarding {topic} necessitates immediate attention from {department} to address {challenge}.",
                "Following our discussion about {topic}, the team should prioritize {action} to maintain progress toward our quarterly goals.",
                "To clarify the decision made during {meeting}, {decision} will be implemented effective {date}.",
                "Based on feedback from {stakeholder}, we need to adjust our approach to {topic} by incorporating {change}."
            ]
        },
        'conclusion': {
            'report': [
                "In conclusion, our analysis of {topic} demonstrates {finding}. We recommend {recommendation} to address these findings and improve overall performance.",
                "Based on the data presented in this report, we conclude that {finding}. The recommended next steps include {recommendation} to leverage these insights.",
                "This report highlights significant {finding} related to {topic}. We advise implementing {recommendation} to capitalize on opportunities and mitigate risks."
            ],
            'policy': [
                "This policy will be reviewed annually and updated as necessary to reflect changes in business needs, technology, or regulatory requirements.",
                "All employees are expected to familiarize themselves with this policy and incorporate these guidelines into their daily activities related to {topic}.",
                "Questions regarding the interpretation or implementation of this policy should be directed to {department} for clarification and guidance."
            ],
            'proposal': [
                "We request approval to proceed with this proposal for {topic}, which requires {resource} to implement. The expected ROI is {metric} within {timeframe}.",
                "Implementation of this proposal will position our organization to {benefit}, creating competitive advantage and supporting our strategic objectives.",
                "We recommend moving forward with this initiative based on the clear {benefit} it offers and its alignment with our organizational priorities for {timeframe}."
            ],
            'analysis': [
                "This analysis reveals important insights about {topic} that should inform our strategic decision-making. Key recommendations include {recommendation}.",
                "Based on our findings, we recommend {recommendation} to leverage the opportunities identified in this analysis and address potential challenges.",
                "The patterns and trends highlighted in this document suggest that {recommendation} would optimize our approach to {topic} moving forward."
            ],
            'memo': [
                "Please direct any questions or concerns about this update to {contact} by {deadline}.",
                "We appreciate your prompt attention to this matter and your ongoing commitment to {value}.",
                "Thank you for your cooperation in implementing these changes to support our organizational objectives."
            ]
        }
    }
    
    # Author templates
    authors = [
        "Alex Johnson, {department} Director",
        "Jamie Smith, Senior {department} Analyst",
        "Taylor Garcia, {department} Manager",
        "Morgan Brown, Chief {department} Officer",
        "Casey Wilson, {department} Specialist",
        "Jordan Lee, {department} Consultant",
        "Riley Jackson, Head of {department}",
        "Quinn Martinez, {department} Lead",
        "Avery Robinson, {department} Coordinator",
        "Skyler Thompson, {department} Executive"
    ]
    
    # Department names
    departments = [
        "Finance", "Marketing", "Operations", "Human Resources", "Sales",
        "Research", "Development", "Strategy", "Technology", "Product"
    ]
    
    # Project names
    projects = [
        "Phoenix", "Horizon", "Nexus", "Catalyst", "Quantum",
        "Atlas", "Pinnacle", "Spectrum", "Velocity", "Fusion"
    ]
    
    # Topics
    topics = [
        "Digital Transformation", "Market Expansion", "Customer Experience",
        "Operational Efficiency", "Talent Development", "Product Innovation",
        "Risk Management", "Sustainability", "Strategic Partnerships",
        "Technology Integration", "Data Security", "Regulatory Compliance"
    ]
    
    # Metrics
    metrics = [
        "Revenue Growth", "Customer Satisfaction", "Employee Engagement",
        "Market Share", "Cost Reduction", "Productivity", "Quality Score",
        "Return on Investment", "User Adoption", "Conversion Rate"
    ]
    
    # Findings/patterns
    findings = [
        "a significant improvement", "a concerning decline", "a stable trend",
        "unexpected variations", "consistent performance", "remarkable growth",
        "periodic fluctuations", "a correlation with external factors",
        "divergent results across segments", "opportunity for optimization"
    ]
    
    patterns = [
        "an upward trajectory", "a downward trend", "cyclical patterns",
        "seasonal variations", "a plateau effect", "exponential growth",
        "diminishing returns", "consistent stability", "irregular fluctuations",
        "a tipping point"
    ]
    
    # Changes for body templates
    changes = [
        "more targeted messaging",
        "streamlined processes",
        "additional validation steps",
        "enhanced reporting mechanisms",
        "cross-functional input"
    ]
    
    # Generate date range
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    date_range_days = (end_date - start_date).days
    
    # Generate documents
    documents = []
    
    # Create output folder if saving files
    output_folder = Path(output_folder)
    if save_files:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_documents):
        # Select document category
        category = random.choice(categories)
        
        # Generate document date
        days_offset = random.randint(0, date_range_days)
        doc_date = start_date + pd.Timedelta(days=days_offset)
        
        # Format document date
        doc_date_str = doc_date.strftime("%B %d, %Y")
        
        # Generate title
        title_template = random.choice(title_templates[category])
        
        # Department for this document
        document_dept = random.choice(departments)
        
        # Format title with only valid placeholders
        format_args = {
            'quarter': f"Q{random.randint(1, 4)}",
            'year': random.randint(2024, 2025),
            'department': document_dept,
            'project': random.choice(projects),
            'topic': random.choice(topics)
        }
        placeholders = re.findall(r'\{([^}]+)\}', title_template)
        title = title_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
        
        # Generate author
        author_template = random.choice(authors)
        author = author_template.format(department=document_dept)
        
        # Determine document length
        min_paragraphs, max_paragraphs = lengths.get(category, (5, 15))
        num_paragraphs = random.randint(min_paragraphs, max_paragraphs)
        
        # Generate document content
        content = []
        
        # Add header
        content.append(f"{title}")
        content.append(f"Author: {author}")
        content.append(f"Date: {doc_date_str}")
        content.append("")  # Blank line
        
        # Add introduction (1-2 paragraphs)
        content.append("INTRODUCTION")
        content.append("")
        
        intro_template = random.choice(content_templates['introduction'][category])
        format_args = {
            'topic': random.choice(topics),
            'period': f"Q{random.randint(1, 4)} {random.randint(2024, 2025)}",
            'objective': f"improving {random.choice(metrics).lower()}",
            'challenge': f"addressing {random.choice(['declining', 'stagnant', 'inconsistent', 'underperforming'])} {random.choice(metrics).lower()}"
        }
        placeholders = re.findall(r'\{([^}]+)\}', intro_template)
        intro = intro_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
        content.append(intro)
        content.append("")
        
        # Add second intro paragraph sometimes
        if random.random() < 0.7:
            intro2_template = random.choice(content_templates['introduction'][category])
            format_args = {
                'topic': random.choice(topics),
                'period': f"Q{random.randint(1, 4)} {random.randint(2024, 2025)}",
                'objective': f"enhancing {random.choice(metrics).lower()}",
                'challenge': f"overcoming {random.choice(['challenges in', 'obstacles to', 'limitations of', 'constraints on'])} {random.choice(metrics).lower()}"
            }
            placeholders = re.findall(r'\{([^}]+)\}', intro2_template)
            intro2 = intro2_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
            content.append(intro2)
            content.append("")
        
        # Add body (variable paragraphs)
        content.append("ANALYSIS & FINDINGS")
        content.append("")
        
        # Generate multiple body sections with headings
        num_body_sections = random.randint(2, 4)
        
        for section in range(num_body_sections):
            # Add section header
            section_topic = random.choice(topics)
            content.append(f"{section_topic.upper()}")
            content.append("")
            
            # Add paragraphs for this section
            section_paragraphs = random.randint(1, 3)
            
            for p in range(section_paragraphs):
                body_template = random.choice(content_templates['body'][category])
                format_args = {
                    'topic': section_topic,
                    'metric': random.choice(metrics),
                    'finding': random.choice(findings),
                    'pattern': random.choice(patterns),
                    'period': f"Q{random.randint(1, 4)} {random.randint(2024, 2025)}",
                    'implication': random.choice([
                        "a need to reconsider our strategy",
                        "an opportunity for growth",
                        "potential risks that require mitigation",
                        "competitive advantages we should leverage",
                        "operational inefficiencies to address"
                    ]),
                    'change': random.choice(changes),
                    'requirement': random.choice([
                        "strict documentation procedures",
                        "regular compliance audits",
                        "proper authorization protocols",
                        "secure handling practices",
                        "standardized reporting methods"
                    ]),
                    'regulation': random.choice([
                        "GDPR", "HIPAA", "SOX", "ISO 27001", "PCI DSS"
                    ]),
                    'approach': random.choice([
                        "a phased implementation plan",
                        "cross-functional collaboration",
                        "technology-enabled automation",
                        "strategic partnership development",
                        "customer-centric redesign"
                    ]),
                    'benefit': random.choice([
                        "cost savings of 15-20%",
                        "improved customer satisfaction",
                        "reduced processing time by 30%",
                        "enhanced data accuracy",
                        "increased employee productivity"
                    ]),
                    'timeframe': random.choice([
                        "the next quarter",
                        "6-8 months",
                        "the current fiscal year",
                        "18 months",
                        "the next two quarters"
                    ]),
                    'variable1': random.choice(metrics),
                    'variable2': random.choice(metrics),
                    'dataset': random.choice([
                        "customer transaction history",
                        "employee performance metrics",
                        "market research findings",
                        "operational efficiency indicators",
                        "financial performance data"
                    ]),
                    'dimension': random.choice([
                        "geographic regions",
                        "customer segments",
                        "product categories",
                        "time periods",
                        "distribution channels"
                    ]),
                    'category': random.choice([
                        "customer",
                        "product",
                        "regional",
                        "departmental",
                        "market"
                    ]),
                    'aspect': random.choice([
                        "operational efficiency",
                        "customer engagement",
                        "market penetration",
                        "cost structure",
                        "quality metrics"
                    ]),
                    'department': document_dept,
                    'action': random.choice([
                        "immediate attention",
                        "collaborative planning",
                        "resource reallocation",
                        "strategic reconsideration",
                        "process optimization"
                    ]),
                    'deadline': random.choice([
                        "the end of the month",
                        "next quarter",
                        "year-end",
                        "within 30 days",
                        "before the next review cycle"
                    ]),
                    'meeting': random.choice([
                        "quarterly review",
                        "strategic planning session",
                        "executive committee meeting",
                        "project kickoff",
                        "team retrospective"
                    ]),
                    'decision': random.choice([
                        "the new approval workflow",
                        "revised budget allocations",
                        "updated project timeline",
                        "modified resource distribution",
                        "refined success metrics"
                    ]),
                    'stakeholder': random.choice([
                        "the leadership team",
                        "key clients",
                        "department managers",
                        "external consultants",
                        "industry partners"
                    ]),
                    'component': random.choice([
                        "automated workflow integration",
                        "predictive analytics capabilities",
                        "customer self-service features",
                        "centralized data management",
                        "real-time performance monitoring"
                    ]),
                    'differentiator': random.choice([
                        "its innovative methodology",
                        "a more comprehensive approach",
                        "superior integration capabilities",
                        "lower implementation costs",
                        "faster time to value"
                    ]),
                    'authority': random.choice([
                        "the Executive Committee",
                        "the Compliance Officer",
                        "the Department Director",
                        "the Security Council",
                        "the Board of Directors"
                    ]),
                    'consequence': random.choice([
                        "disciplinary action",
                        "compliance violations",
                        "security incidents",
                        "legal liability",
                        "operational disruption"
                    ]),
                    'objective': random.choice([
                        "protecting sensitive information",
                        "maintaining regulatory compliance",
                        "ensuring operational consistency",
                        "promoting ethical standards",
                        "safeguarding organizational assets"
                    ]),
                    'challenge': f"addressing {random.choice(['declining', 'stagnant', 'inconsistent', 'underperforming'])} {random.choice(metrics).lower()}"
                }
                placeholders = re.findall(r'\{([^}]+)\}', body_template)
                body = body_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
                content.append(body)
                content.append("")
        
        # Add conclusion (1-2 paragraphs)
        content.append("CONCLUSION & RECOMMENDATIONS")
        content.append("")
        
        conclusion_template = random.choice(content_templates['conclusion'][category])
        format_args = {
            'topic': random.choice(topics),
            'finding': random.choice(findings),
            'recommendation': random.choice([
                "implementing a phased approach to digital transformation",
                "investing in employee training and development",
                "revising our market segmentation strategy",
                "optimizing our operational workflow",
                "enhancing our data analytics capabilities"
            ]),
            'resource': random.choice([
                "an initial investment of $100,000-150,000",
                "dedicated cross-functional team resources",
                "technology infrastructure upgrades",
                "specialized consulting expertise",
                "reallocation of existing departmental budgets"
            ]),
            'metric': random.choice([
                "20-30% increase in efficiency",
                "15% reduction in operational costs",
                "25% improvement in customer satisfaction",
                "doubling of conversion rates",
                "40% decrease in processing time"
            ]),
            'timeframe': random.choice([
                "6-8 months",
                "the next fiscal year",
                "two quarters",
                "12-18 months",
                "the current annual cycle"
            ]),
            'benefit': random.choice([
                "strengthen our competitive position",
                "enhance operational efficiency",
                "increase market share",
                "improve customer loyalty",
                "reduce overhead costs"
            ]),
            'department': document_dept,
            'contact': f"{random.choice(['Alex', 'Jamie', 'Taylor', 'Morgan', 'Casey'])} at {random.choice(['x123', 'x456', 'x789'])}",
            'deadline': random.choice([
                "end of week",
                "close of business Friday",
                "our next team meeting",
                "the 15th of this month",
                "the quarterly review"
            ]),
            'value': random.choice([
                "operational excellence",
                "customer satisfaction",
                "innovation",
                "teamwork",
                "continuous improvement"
            ])
        }
        placeholders = re.findall(r'\{([^}]+)\}', conclusion_template)
        conclusion = conclusion_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
        content.append(conclusion)
        content.append("")
        
        # Add second conclusion paragraph sometimes
        if random.random() < 0.5 and category != 'memo':
            conclusion2_template = random.choice(content_templates['conclusion'][category])
            format_args = {
                'topic': random.choice(topics),
                'finding': random.choice(findings),
                'recommendation': random.choice([
                    "realigning our resource allocation to prioritize high-impact initiatives",
                    "establishing clear metrics for tracking progress and outcomes",
                    "developing a communication strategy to ensure stakeholder alignment",
                    "creating a governance framework for ongoing monitoring and adjustment",
                    "building strategic partnerships to extend our capabilities"
                ]),
                'resource': random.choice([
                    "phased funding of $50,000 per quarter",
                    "two dedicated full-time resources",
                    "implementation of specialized software tools",
                    "executive sponsorship and oversight",
                    "training and change management support"
                ]),
                'metric': random.choice([
                    "ROI of 200-250% within eighteen months",
                    "35% increase in team productivity",
                    "50% reduction in error rates",
                    "tripling of our innovation pipeline",
                    "10% increase in market penetration"
                ]),
                'timeframe': random.choice([
                    "the remainder of this year",
                    "a three-quarter implementation period",
                    "the next two fiscal cycles",
                    "our five-year strategic plan",
                    "an accelerated six-month timeline"
                ]),
                'benefit': random.choice([
                    "drive sustainable growth",
                    "create significant competitive differentiation",
                    "establish industry leadership",
                    "transform our business model",
                    "achieve breakthrough performance"
                ]),
                'department': document_dept,
                'contact': f"{random.choice(['Riley', 'Quinn', 'Avery', 'Skyler', 'Jordan'])} at {random.choice(['x234', 'x567', 'x890'])}",
                'deadline': random.choice([
                    "month-end",
                    "the executive retreat",
                    "the department all-hands",
                    "the start of next quarter",
                    "the project kickoff meeting"
                ]),
                'value': random.choice([
                    "strategic focus",
                    "accountability",
                    "data-driven decision making",
                    "customer centricity",
                    "agility and adaptability"
                ])
            }
            placeholders = re.findall(r'\{([^}]+)\}', conclusion2_template)
            conclusion2 = conclusion2_template.format(**{k: v for k, v in format_args.items() if k in placeholders})
            content.append(conclusion2)
            content.append("")
        
        # Join content
        document_text = "\n".join(content)
        
        # Save document to file if requested
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().lower()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"doc_{i+1:03d}_{safe_title[:30]}.txt"
        filepath = output_folder / filename
        size_kb = len(document_text.encode('utf-8')) / 1024  # Calculate size in KB
        
        if save_files:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(document_text)
                size_kb = filepath.stat().st_size / 1024  # Update with actual file size
            except Exception as e:
                print(f"Error saving {filename}: {e}")
        
        # Calculate document statistics
        word_count = len(document_text.split())
        char_count = len(document_text)
        line_count = document_text.count('\n') + 1
        
        # Add to documents list
        document_metadata = {
            'filename': filename,
            'path': str(filepath),
            'extension': 'txt',
            'size_kb': size_kb,
            'created_date': doc_date,
            'modified_date': doc_date,  # Assume same as created for synthetic docs
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'content': document_text,
            'title': title,
            'category': category,
            'author': author,
            'department': document_dept,
            'n_paragraphs': num_paragraphs
        }
        
        documents.append(document_metadata)
    
    # Create DataFrame from documents
    documents_df = pd.DataFrame(documents)
    
    # Save metadata to CSV if saving files
    if save_files:
        metadata_file = output_folder / 'document_metadata.csv'
        documents_df.to_csv(metadata_file, index=False)
        print(f"Generated {n_documents} synthetic documents in '{output_folder}'")
        print(f"Metadata saved to '{metadata_file}'")
    
    return documents_df

def generate_synthetic_text_dataset(n_samples: int = 100,
                                  categories: Optional[List[str]] = None,
                                  lengths: Optional[Tuple[int, int]] = None,
                                  output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a synthetic text classification dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    categories : Optional[List[str]]
        List of categories (if None, uses predefined categories)
    lengths : Optional[Tuple[int, int]]
        Range of text lengths (min, max) in words
    output_file : Optional[str]
        Path to save the dataset CSV (if None, doesn't save)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with generated text samples and categories
    """
    if categories is None:
        categories = ['technology', 'business', 'sports', 'health', 'entertainment']
    
    if lengths is None:
        lengths = (50, 150)
    
    # Templates for different categories
    templates = {
        'technology': [
            "The latest advancements in {tech_field} technology are transforming how companies approach {tech_application}. Experts at {tech_company} have developed new {tech_product} that {tech_benefit}. According to {tech_person}, {quote_tech}. This innovation comes at a time when the industry is focusing on {tech_trend}. Many organizations are investing in these solutions to {tech_goal}. The market for {tech_field} is expected to grow by {percent}% in the next {timeframe}, reaching {amount} by {year}. Critics argue that {tech_concern}, but proponents maintain that {tech_advantage}. As more businesses adopt these technologies, we can expect to see {tech_outcome}.",
            "Researchers at {tech_company} have announced a breakthrough in {tech_field} that could revolutionize {tech_application}. The new {tech_product} utilizes {tech_approach} to {tech_benefit}. During testing, the technology demonstrated {tech_metric} improvement over existing solutions. {tech_person}, who led the research team, explained, \"{quote_tech}\" This development addresses long-standing challenges in {tech_challenge}. Industry analysts predict this could disrupt {tech_market} by enabling companies to {tech_goal}. Early adopters include {tech_adopter}, who are already implementing the technology to {tech_use_case}. However, questions remain about {tech_concern} that developers will need to address before widespread adoption occurs.",
            "The intersection of {tech_field1} and {tech_field2} is creating new opportunities for innovation in {tech_application}. Start-up {tech_company} has secured {amount} in funding to develop {tech_product} that combines both technologies. The solution promises to {tech_benefit} while addressing {tech_challenge}. {tech_person} of {analyst_firm} notes, \"{quote_tech}\" Companies implementing similar hybrid approaches have reported {tech_metric} improvements in operational efficiency. The technology leverages {tech_approach} to overcome traditional limitations in {tech_limitation}. Despite enthusiasm, industry experts caution that {tech_concern}. The next {timeframe} will be critical as these technologies mature and find their place in the {tech_market} ecosystem."
        ],
        'business': [
            "The {business_sector} industry is experiencing significant changes due to {business_factor}. Companies like {business_company} are implementing new strategies to {business_goal}. According to recent reports, the market has seen {percent}% growth in {timeframe}, with projections indicating continued expansion. {business_person}, {business_title} at {analyst_firm}, states that \"{quote_business}\" This trend has implications for {business_stakeholder}, who must adapt to {business_change}. Some organizations have already invested in {business_investment} to stay competitive. Analysts recommend businesses focus on {business_strategy} to navigate the evolving landscape. Meanwhile, concerns about {business_concern} remain a challenge that industry leaders are working to address.",
            "A new study by {analyst_firm} reveals that {business_sector} companies embracing {business_trend} outperform their peers by {percent}%. {business_company} exemplifies this approach, having {business_achievement} after implementing a comprehensive {business_strategy} initiative. {business_person}, {business_title}, explained their success: \"{quote_business}\" The research identified key factors driving this performance gap, including {business_factor1}, {business_factor2}, and {business_factor3}. Organizations slow to adopt these practices reported struggling with {business_challenge}. Industry experts recommend a phased approach to implementation, starting with {business_starting_point}. The economic outlook suggests that {business_prediction}, making these adaptations increasingly important for long-term viability.",
            "Merger and acquisition activity in the {business_sector} sector has increased by {percent}% this quarter, with {business_company1} announcing its {amount} acquisition of {business_company2}. This consolidation reflects broader trends in the industry, as companies seek to {business_goal}. Market analysts at {analyst_firm} suggest this deal will {business_impact}. {business_person}, who advised on the transaction, noted that \"{quote_business}\" The combined entity will control approximately {percent}% of the market share, raising questions about {business_concern}. Competitors are responding by {business_response}. Customers may experience {business_customer_impact} as the integration progresses over the next {timeframe}. Regulatory authorities are expected to {regulatory_action} given the scope of the deal."
        ],
        'sports': [
            "The {sports_team} dominated their match against {sports_opponent}, winning by {score}. {sports_player}, who {player_achievement}, was named player of the game. Coach {sports_coach} attributed the victory to {sports_strategy}, saying \"{quote_sports}\" The team has now won {number} consecutive games, placing them at the top of the {sports_league} standings. Analysts note that their improvement in {sports_skill} has been a key factor in their recent success. However, concerns about {sports_concern} could affect their performance in upcoming matches against {sports_nextopponent}. Fans are optimistic about the team's prospects for the playoffs, with ticket sales increasing by {percent}% compared to last season.",
            "In a surprising development, {sports_player} has announced {player_announcement} after {number} seasons with {sports_team}. The {player_position} has been instrumental in the team's success, including {sports_achievement}. Team management expressed {reaction} in a statement: \"{quote_sports}\" Industry insiders suggest that {sports_rumor} may have influenced the decision. This change comes at a critical time in the {sports_league} season, with {sports_team} currently {team_standing}. Analysts from {sports_media} predict that this will impact the team's {sports_aspect}, potentially affecting their championship aspirations. Meanwhile, supporters have reacted with {fan_reaction}, with many taking to social media to share their thoughts on this significant roster change.",
            "The upcoming {sports_event} is generating excitement as {sports_player1} prepares to face {sports_player2} in what experts are calling a historic matchup. {sports_player1}, known for {player1_strength}, enters the competition after {player1_recent}. Meanwhile, {sports_player2} has demonstrated exceptional {player2_strength} throughout this season. {sports_commentator} of {sports_media} observes, \"{quote_sports}\" Bookmakers have established {sports_player1} as the {odds} favorite, though many analysts point to {factor} as a potential equalizer. The venue, {sports_venue}, has been prepared with {venue_feature} to accommodate the anticipated record attendance. This contest could determine {sports_stake}, adding to the significance of the event."
        ],
        'health': [
            "A new study published in {health_journal} has found that {health_finding} related to {health_topic}. Researchers at {health_institution} conducted a {study_type} study involving {number} participants over {timeframe}. The results showed {health_result}, which could have implications for {health_application}. Dr. {health_researcher}, lead author of the study, explained that \"{quote_health}\" Health experts recommend that individuals should {health_recommendation} based on these findings. However, some specialists, including Dr. {health_critic}, caution that {health_limitation}. The research team plans to {next_steps} to address unanswered questions about {health_question}.",
            "Health officials are reporting a {percent}% {increase_decrease} in cases of {health_condition} compared to last year. The trend is particularly notable in {health_demographic}, where rates have {health_rate_change}. Dr. {health_official} of {health_organization} attributes this change to {health_factor}, stating \"{quote_health}\" Prevention efforts have focused on {health_prevention}, which has shown promise in {health_outcome}. Community health workers are implementing {health_initiative} to reach vulnerable populations. Experts emphasize the importance of {health_practice} in managing personal risk. Looking ahead, health authorities predict {health_prediction} if current patterns continue, highlighting the need for coordinated public health responses and increased awareness.",
            "A breakthrough in {health_field} treatment offers new hope for patients with {health_condition}. The {health_treatment}, developed by researchers at {health_institution}, works by {treatment_mechanism}. Clinical trials involving {number} patients demonstrated a {percent}% improvement in {health_metric}, with {health_side_effect} reported in only {small_percent}% of cases. Dr. {health_researcher} describes the advance as \"{quote_health}\" This approach represents a departure from conventional treatments that {conventional_limitation}. Insurance coverage for the new therapy {insurance_status}, which has raised concerns about {health_access_issue}. Patient advocacy groups like {health_organization} are working to {advocacy_goal} to ensure broader access to this promising intervention."
        ],
        'entertainment': [
            "The highly anticipated film \"{entertainment_title}\" has exceeded expectations, grossing {amount} in its opening weekend. Directed by {entertainment_director} and starring {entertainment_star}, the {entertainment_genre} film has received {rating} from critics. Reviewers have particularly praised the {film_element}, with {entertainment_critic} of {entertainment_outlet} writing that \"{quote_entertainment}\" Audiences have responded positively to the {film_aspect}, contributing to its {percent}% approval rating on review aggregator sites. The success comes despite {entertainment_challenge} during production. Industry analysts predict the film will reach {amount_total} in total box office revenue, potentially spawning a franchise. A sequel has already been {sequel_status}, with {entertainment_star} expected to reprise their role.",
            "Grammy-winning artist {entertainment_star} has released a surprise new album titled \"{entertainment_title}.\" The {music_genre} project features collaborations with {entertainment_collaborator} and production by {entertainment_producer}. Critics have noted the album's {music_element}, with {entertainment_critic} describing it as \"{quote_entertainment}\" This marks a {music_direction} for the artist, who previously {artist_history}. The lead single, \"{song_title},\" has already reached {chart_position} on the Billboard charts and accumulated {number} streams on digital platforms. Fans have expressed {fan_reaction} to this unexpected release, particularly praising tracks like \"{track_name}\" for its {track_quality}. A supporting tour has been announced to begin in {timeframe}.",
            "The latest season of \"{entertainment_title}\" has become a cultural phenomenon, breaking streaming records with {number} viewers in its first {timeframe}. The {entertainment_genre} series, created by {entertainment_creator}, has generated significant online discussion about its {show_element}. {entertainment_critic} of {entertainment_outlet} noted that \"{quote_entertainment}\" Social media analysis reveals {number_high} mentions of the show's {viral_element}, which has spawned numerous memes and fan theories. The cast, including {entertainment_star}, has been praised for {performance_quality}. Industry insiders report that the series has boosted subscriber numbers for {streaming_service} by {percent}%. Production of the next season is {production_status}, with a projected release date in {release_timeframe}."
        ]
    }
    
    # Category-specific word banks
    word_banks = {
        'technology': {
            'tech_field': ['artificial intelligence', 'blockchain', 'quantum computing', 'augmented reality', 
                          'machine learning', 'edge computing', 'internet of things', 'cybersecurity',
                          'robotics', 'biotechnology', '5G', 'cloud computing'],
            'tech_field1': ['artificial intelligence', 'blockchain', 'quantum computing', 'augmented reality'],
            'tech_field2': ['machine learning', 'edge computing', 'internet of things', 'cybersecurity'],
            'tech_application': ['healthcare', 'finance', 'manufacturing', 'retail', 'transportation',
                               'education', 'agriculture', 'energy management', 'customer service',
                               'supply chain optimization', 'urban planning', 'security'],
            'tech_company': ['Google', 'Microsoft', 'Apple', 'Amazon', 'Tesla', 'IBM', 'Intel',
                           'Samsung', 'NVIDIA', 'Oracle', 'Cisco', 'Salesforce', 'Siemens',
                           'TechCorp', 'InnovateSoft', 'NextGen Systems', 'Quantum Dynamics',
                           'Apex Technologies', 'ByteWave', 'CoreTech Solutions'],
            'tech_product': ['platform', 'system', 'device', 'framework', 'algorithm', 'tool',
                           'application', 'interface', 'protocol', 'architecture', 'solution',
                           'processor', 'infrastructure', 'network'],
            'tech_benefit': ['increases efficiency by 40%', 'reduces operational costs significantly',
                           'enhances data security', 'improves accuracy by 60%', 'streamlines complex processes',
                           'enables real-time decision making', 'minimizes human error', 'accelerates development cycles',
                           'optimizes resource allocation', 'provides unprecedented scalability',
                           'delivers actionable insights', 'transforms user experience'],
            'tech_person': ['Dr. Sarah Chen', 'Professor James Rodriguez', 'CTO Michael Wong',
                          'Research Director Emily Johnson', 'Chief Scientist David Kim',
                          'Lead Engineer Priya Patel', 'Innovation Officer Thomas Wilson',
                          'Technology Strategist Rebecca Martinez', 'Principal Researcher Alex Taylor',
                          'Systems Architect Raj Mehta'],
            'quote_tech': ['this breakthrough represents a paradigm shift in how we approach these problems',
                         'we\'re finally seeing the convergence of multiple technologies to solve real-world challenges',
                         'the implications for industry and society are profound and far-reaching',
                         'after years of research, we\'ve achieved what many thought was impossible',
                         'this technology addresses the fundamental limitations that have held back progress',
                         'we\'ve essentially eliminated the trade-off between performance and efficiency',
                         'our approach fundamentally rethinks how these systems should be designed'],
            'tech_trend': ['sustainability', 'ethical AI', 'decentralized systems', 'privacy preservation',
                         'edge intelligence', 'autonomous operations', 'human-computer collaboration',
                         'predictive analytics', 'zero-trust security', 'digital twins'],
            'tech_goal': ['gain competitive advantage', 'modernize legacy systems', 'enhance customer experiences',
                        'reduce environmental impact', 'comply with regulatory requirements',
                        'accelerate digital transformation', 'enable data-driven decision making',
                        'improve operational resilience', 'create new business models'],
            'tech_concern': ['privacy implications remain a significant concern', 'the technology raises important ethical questions',
                           'security vulnerabilities could pose substantial risks', 'the environmental footprint is still problematic',
                           'accessibility issues may limit widespread adoption', 'integration with existing systems presents challenges',
                           'the skills gap could impede implementation', 'regulatory uncertainty creates compliance challenges'],
            'tech_advantage': ['the benefits far outweigh the potential risks', 'proper governance frameworks can address most concerns',
                             'the technology includes built-in safeguards', 'these issues are addressable through iterative improvement',
                             'transparent development practices mitigate many problems', 'extensive testing has validated safety protocols'],
            'tech_outcome': ['widespread disruption across multiple industries', 'new categories of jobs and business opportunities',
                           'greater accessibility to previously exclusive capabilities', 'a fundamental shift in consumer expectations',
                           'significant improvements in quality of life', 'accelerated innovation in adjacent fields',
                           'more sustainable and efficient operations'],
            'tech_approach': ['neural network architecture', 'distributed ledger technology', 'quantum algorithms',
                            'federated learning', 'generative modeling', 'edge processing', 'natural language understanding',
                            'computer vision', 'biometric authentication', 'predictive modeling'],
            'tech_metric': ['a 200%', 'an 80%', 'a 125%', 'a 3x', 'a substantial', 'a significant',
                          'an unprecedented', 'a remarkable', 'a consistent'],
            'tech_challenge': ['data privacy', 'processing speed', 'energy consumption', 'algorithm bias',
                             'integration complexity', 'reliability at scale', 'security vulnerabilities',
                             'interoperability', 'skill requirements', 'implementation costs'],
            'tech_market': ['financial services', 'healthcare delivery', 'retail operations', 'manufacturing processes',
                          'supply chain management', 'educational technology', 'urban infrastructure',
                          'transportation systems', 'energy distribution', 'agricultural production'],
            'tech_adopter': ['JP Morgan Chase', 'Mayo Clinic', 'Walmart', 'Toyota', 'DHL',
                           'Stanford University', 'Singapore\'s Smart Nation initiative', 'Delta Airlines',
                           'National Grid', 'Archer Daniels Midland', 'Global Manufacturing Inc.',
                           'Metropolitan Health System', 'EduTech University'],
            'tech_use_case': ['optimize trading algorithms', 'improve diagnostic accuracy', 'enhance inventory management',
                             'streamline production processes', 'reduce delivery times', 'personalize learning experiences',
                             'improve traffic flow', 'balance load distribution', 'increase crop yields',
                             'transform customer interactions'],
            'tech_limitation': ['real-time processing', 'data integration', 'system interoperability',
                              'scalability', 'error handling', 'resource utilization', 'user adoption',
                              'security protocols', 'cross-platform compatibility']
        },
        'business': {
            'business_sector': ['retail', 'financial services', 'healthcare', 'manufacturing', 'technology',
                              'transportation', 'hospitality', 'energy', 'telecommunications', 'real estate',
                              'professional services', 'media and entertainment'],
            'business_factor': ['changing consumer preferences', 'technological disruption', 'regulatory changes',
                              'supply chain challenges', 'labor market dynamics', 'economic uncertainty',
                              'sustainability demands', 'digital transformation', 'globalization',
                              'increasing competition'],
            'business_company': ['Amazon', 'JPMorgan Chase', 'Cleveland Clinic', 'Toyota', 'Microsoft',
                               'FedEx', 'Marriott International', 'ExxonMobil', 'Verizon', 'Prologis',
                               'McKinsey & Company', 'Disney', 'Global Enterprises Inc.',
                               'Meridian Solutions', 'Apex Industries', 'Summit Financial Group',
                               'Cornerstone Healthcare', 'Nova Technologies'],
            'business_company1': ['Berkshire Hathaway', 'Goldman Sachs', 'CitiGroup', 'Johnson & Johnson',
                                'Pfizer', 'Procter & Gamble', 'Unilever', 'General Electric',
                                'Siemens', 'BP', 'Shell', 'AT&T', 'Comcast', 'Morgan Stanley'],
            'business_company2': ['Whole Foods', 'E*Trade', 'WebMD', 'SmithCorp Manufacturing',
                                'Salesforce', 'TNT Express', 'Hyatt Hotels', 'Phillips 66',
                                'Sprint', 'WeWork', 'Boston Consulting Group', 'Time Warner',
                                'Regional Leaders', 'Quantum Financial', 'MediCare Systems',
                                'Precision Manufacturing', 'Digital Innovations'],
            'business_goal': ['enhance customer experience', 'reduce operational costs', 'expand market share',
                            'accelerate digital transformation', 'increase operational efficiency',
                            'develop new revenue streams', 'strengthen competitive position',
                            'improve supply chain resilience', 'attract and retain talent',
                            'meet ESG commitments'],
            'business_person': ['Sarah Johnson', 'Michael Chen', 'David Rodriguez', 'Lisa Patel',
                              'Robert Kim', 'Jennifer Williams', 'Carlos Mendez', 'Emma Wilson',
                              'James Thompson', 'Maria Garcia'],
            'business_title': ['CEO', 'Chief Financial Officer', 'Chief Strategy Officer', 'Director of Operations',
                             'Senior Vice President', 'Managing Partner', 'Head of Innovation',
                             'Chief Marketing Officer', 'Global Supply Chain Director',
                             'VP of Human Resources', 'Chief Sustainability Officer',
                             'Digital Transformation Lead', 'Senior Market Analyst'],
            'analyst_firm': ['McKinsey & Company', 'Boston Consulting Group', 'Deloitte', 'PwC',
                           'Gartner', 'Forrester Research', 'Bain & Company', 'KPMG',
                           'Ernst & Young', 'Goldman Sachs', 'Morgan Stanley', 'JP Morgan',
                           'Accenture', 'IDC', 'Constellation Research'],
            'quote_business': ['companies that fail to adapt to this shift will find themselves at a significant competitive disadvantage',
                             'the data clearly shows that organizations embracing this change outperform their peers by every key metric',
                             'we\'re seeing unprecedented opportunities for businesses that can effectively navigate this transition',
                             'successful implementation requires alignment across strategy, technology, and organizational culture',
                             'this represents a fundamental reimagining of traditional business models in the sector',
                             'the most successful companies are those that view this not as a challenge but as a strategic opportunity'],
            'business_stakeholder': ['shareholders', 'employees', 'customers', 'suppliers', 'regulators',
                                   'community partners', 'industry associations', 'investors',
                                   'strategic partners', 'market analysts'],
            'business_change': ['evolving consumer expectations', 'technological innovation', 'new competitive landscapes',
                              'increasing regulatory scrutiny', 'changing workforce dynamics',
                              'volatile market conditions', 'sustainability imperatives',
                              'supply chain reconfiguration', 'digital-first business models'],
            'business_investment': ['advanced analytics capabilities', 'digital infrastructure', 'workforce development',
                                  'sustainable operations', 'customer experience platforms',
                                  'supply chain optimization tools', 'automation technologies',
                                  'strategic acquisitions', 'innovation labs'],
            'business_strategy': ['customer-centric transformation', 'data-driven decision making', 'agile operating models',
                                'strategic partnerships', 'sustainable business practices',
                                'value chain optimization', 'talent development and retention',
                                'digital-first capabilities', 'innovation ecosystems'],
            'business_concern': ['economic uncertainty', 'regulatory compliance', 'cybersecurity threats',
                               'talent shortages', 'supply chain disruptions', 'margin pressures',
                               'changing consumer preferences', 'competitive intensity',
                               'technological disruption', 'sustainability requirements'],
            'business_trend': ['digital transformation', 'sustainability initiatives', 'agile methodology',
                             'remote work policies', 'customer experience focus', 'data-driven decision making',
                             'supply chain resilience', 'ecosystem partnerships', 'platform business models',
                             'employee well-being programs'],
            'business_achievement': ['increased market share by 15%', 'reduced operational costs by 22%',
                                   'improved customer satisfaction scores by 30 points',
                                   'accelerated time-to-market by 40%', 'boosted employee engagement by 25%',
                                   'achieved carbon neutrality across operations',
                                   'established industry-leading position in key segments',
                                   'delivered record-breaking financial performance'],
            'business_factor1': ['leadership commitment', 'cross-functional collaboration', 'technology enablement'],
            'business_factor2': ['customer-centricity', 'data-driven decision making', 'agile implementation'],
            'business_factor3': ['ecosystem partnerships', 'talent development', 'continuous innovation'],
            'business_challenge': ['siloed organizational structures', 'legacy technology constraints',
                                 'resistance to change', 'skill gaps', 'competing priorities',
                                 'inadequate data infrastructure', 'misaligned incentives',
                                 'insufficient executive sponsorship'],
            'business_starting_point': ['assessing current capabilities', 'defining clear success metrics',
                                      'establishing cross-functional teams', 'implementing pilot programs',
                                      'developing a comprehensive data strategy',
                                      'aligning leadership around key objectives',
                                      'investing in foundational technology'],
            'business_prediction': ['industry consolidation will accelerate', 'customer expectations will continue to evolve rapidly',
                                  'technological disruption will intensify', 'sustainability will become a core business imperative',
                                  'workforce dynamics will undergo fundamental shifts',
                                  'regulatory oversight will increase across sectors',
                                  'new business models will emerge to capture changing value pools'],
            'business_impact': ['create significant synergies in product development', 'reshape competitive dynamics in the sector',
                              'accelerate digital transformation initiatives', 'enhance market access and distribution capabilities',
                              'drive operational efficiencies through scale', 'strengthen resilience against market volatility',
                              'enable more comprehensive customer solutions'],
            'business_response': ['pursuing strategic acquisitions of their own', 'accelerating innovation initiatives',
                                'forming new strategic partnerships', 'doubling down on customer experience differentiation',
                                'optimizing operational efficiency to improve competitiveness',
                                'repositioning their market offerings', 'investing in talent development'],
            'business_customer_impact': ['enhanced product offerings', 'potential pricing changes',
                                       'transitional service disruptions', 'access to a broader ecosystem',
                                       'changes to support structures', 'new terms of service',
                                       'consolidated account management'],
            'regulatory_action': ['conduct an in-depth antitrust review', 'impose certain divestiture requirements',
                                'mandate specific competitive safeguards', 'approve with conditions',
                                'scrutinize data privacy implications', 'evaluate labor market impacts',
                                'assess national security considerations']
        },
        'sports': {
            'sports_team': ['Manchester United', 'New York Yankees', 'Los Angeles Lakers', 'New England Patriots',
                          'Barcelona FC', 'Chicago Bulls', 'Golden State Warriors', 'Real Madrid',
                          'Dallas Cowboys', 'Boston Red Sox', 'Atlanta Braves', 'Bayern Munich',
                          'Toronto Raptors', 'Pittsburgh Steelers', 'Chicago Blackhawks',
                          'Sporting Tigers', 'Metro Knights', 'Capital City Falcons',
                          'Pacific Waves', 'Mountain Lions'],
            'sports_opponent': ['Liverpool FC', 'Boston Red Sox', 'Miami Heat', 'Buffalo Bills',
                              'Real Madrid', 'Detroit Pistons', 'Houston Rockets', 'Atletico Madrid',
                              'Philadelphia Eagles', 'New York Mets', 'Los Angeles Dodgers', 'Borussia Dortmund',
                              'Boston Celtics', 'Baltimore Ravens', 'Detroit Red Wings',
                              'United Eagles', 'City Sharks', 'River Runners',
                              'Atlantic Thunder', 'Midwest Express'],
            'sports_nextopponent': ['Chelsea FC', 'Tampa Bay Rays', 'Brooklyn Nets', 'Kansas City Chiefs',
                                  'Juventus', 'Milwaukee Bucks', 'Phoenix Suns', 'PSG',
                                  'San Francisco 49ers', 'Houston Astros', 'San Francisco Giants', 'RB Leipzig',
                                  'Philadelphia 76ers', 'Cincinnati Bengals', 'Colorado Avalanche',
                                  'Western Wolves', 'Eastern Bears', 'Southern Stars',
                                  'Northern Force', 'Central United'],
            'sports_player': ['Cristiano Ronaldo', 'Aaron Judge', 'LeBron James', 'Tom Brady',
                            'Lionel Messi', 'Giannis Antetokounmpo', 'Kevin Durant', 'Karim Benzema',
                            'Patrick Mahomes', 'Shohei Ohtani', 'Ronald Acuna Jr.', 'Robert Lewandowski',
                            'Kawhi Leonard', 'Lamar Jackson', 'Connor McDavid',
                            'Marcus Williams', 'Carlos Rodriguez', 'Jamal Thompson',
                            'Lisa Chen', 'David Wilson'],
            'sports_player1': ['Rafael Nadal', 'Novak Djokovic', 'Serena Williams', 'Naomi Osaka',
                             'Magnus Carlsen', 'Simone Biles', 'Katie Ledecky', 'Eliud Kipchoge',
                             'Tiger Woods', 'Brooks Koepka', 'Mikaela Shiffrin', 'Sydney McLaughlin',
                             'Tadej Pogačar', 'Kylian Mbappé', 'Virat Kohli'],
            'sports_player2': ['Roger Federer', 'Carlos Alcaraz', 'Iga Swiatek', 'Coco Gauff',
                             'Hikaru Nakamura', 'Sunisa Lee', 'Caeleb Dressel', 'Joshua Cheptegei',
                             'Rory McIlroy', 'Jon Rahm', 'Marco Odermatt', 'Athing Mu',
                             'Jonas Vingegaard', 'Erling Haaland', 'Babar Azam'],
            'score': ['3-1', '7-2', '105-98', '24-17', '2-0', '112-107', '4-2', '1-0', '28-24', '5-3'],
            'sports_coach': ['Jurgen Klopp', 'Aaron Boone', 'Erik Spoelstra', 'Bill Belichick',
                           'Xavi Hernandez', 'Billy Donovan', 'Steve Kerr', 'Carlo Ancelotti',
                           'Mike McCarthy', 'Alex Cora', 'Brian Snitker', 'Julian Nagelsmann',
                           'Nick Nurse', 'Mike Tomlin', 'Paul Maurice',
                           'Sarah Johnson', 'Michael Chen', 'Robert Davis'],
            'sports_strategy': ['disciplined defense', 'aggressive offense', 'strategic substitutions',
                              'tactical flexibility', 'set piece execution', 'fast-break opportunities',
                              'ball movement', 'clock management', 'press resistance',
                              'counterattacking', 'zone coverage', 'man-to-man marking'],
            'quote_sports': ['the team showed incredible character and determination tonight',
                           'we executed our game plan perfectly against a tough opponent',
                           'our preparation and attention to detail made the difference',
                           'this performance demonstrates what we\'re capable of when we play to our potential',
                           'it was a total team effort - everyone contributed to this result',
                           'we\'ve been working on these specific situations in training, and it paid off today'],
            'player_achievement': ['scored the winning goal', 'hit two home runs', 'recorded a triple-double',
                                 'threw for 350 yards and 3 touchdowns', 'completed a hat-trick',
                                 'posted 40 points and 15 rebounds', 'made the decisive 3-pointer',
                                 'saved a penalty kick', 'rushed for 180 yards', 'pitched a complete game',
                                 'broke the course record', 'set a new personal best'],
            'sports_league': ['Premier League', 'Major League Baseball', 'National Basketball Association',
                            'National Football League', 'La Liga', 'Western Conference',
                            'Eastern Conference', 'Champions League', 'NFC East', 'American League East',
                            'National League', 'Bundesliga', 'Atlantic Division', 'AFC North',
                            'Central Division'],
            'sports_skill': ['ball control', 'batting', 'three-point shooting', 'pass completion',
                           'defensive organization', 'rebounding', 'transition offense',
                           'positional awareness', 'red zone efficiency', 'fielding',
                           'pitching', 'finishing', 'perimeter defense', 'offensive line protection',
                           'power play execution'],
            'sports_concern': ['injuries to key players', 'upcoming difficult schedule', 'defensive vulnerabilities',
                             'offensive inconsistency', 'fatigue from the congested fixture list',
                             'integration of new players', 'away form', 'penalty kill effectiveness',
                             'turnover problems', 'late-game execution'],
            'player_announcement': ['their retirement', 'a contract extension', 'a transfer request',
                                  'signing with a rival team', 'moving to a European club',
                                  'taking a coaching role', 'launching a new foundation',
                                  'returning from injury', 'a one-year sabbatical'],
            'player_position': ['striker', 'shortstop', 'point guard', 'quarterback',
                              'midfielder', 'center', 'small forward', 'goalkeeper',
                              'wide receiver', 'pitcher', 'outfielder', 'defender',
                              'power forward', 'offensive tackle', 'goaltender'],
            'sports_achievement': ['winning the championship', 'breaking the all-time scoring record',
                                 'making six consecutive playoff appearances', 'clinching the division title',
                                 'achieving the league\'s best defensive record', 'leading the team to an undefeated season',
                                 'securing promotion to the top division', 'winning the scoring title',
                                 'earning MVP honors', 'reaching the conference finals'],
            'reaction': ['disappointment', 'understanding', 'gratitude for the player\'s contributions',
                       'surprise', 'confidence in the team\'s depth', 'optimism about future prospects',
                       'respect for the player\'s decision', 'determination to maintain momentum'],
            'sports_rumor': ['ongoing contract disputes', 'interest from overseas teams',
                           'disagreements with coaching staff', 'desire to play in a different system',
                           'family considerations', 'pursuit of championship opportunities',
                           'concerns about the team\'s direction', 'injury recovery challenges'],
            'team_standing': ['leading the division', 'fighting for a playoff spot', 'defending their championship title',
                            'in the midst of a rebuilding phase', 'on a record-breaking winning streak',
                            'struggling to find consistent form', 'in second place, just points behind the leaders',
                            'unexpectedly in contention despite preseason predictions'],
            'sports_media': ['ESPN', 'Sky Sports', 'The Athletic', 'NBC Sports',
                           'Sports Illustrated', 'TNT', 'BBC Sport', 'Fox Sports',
                           'CBS Sports', 'Yahoo Sports', 'beIN Sports', 'DAZN',
                           'The Sports Network', 'Stadium', 'Bleacher Report'],
            'sports_aspect': ['offensive production', 'defensive stability', 'locker room chemistry',
                            'tactical approach', 'depth chart', 'salary cap situation',
                            'playoff positioning', 'late-game execution', 'transition game',
                            'special teams performance'],
            'fan_reaction': ['mixed emotions', 'overwhelming support', 'disappointment but understanding',
                           'viral tributes on social media', 'calls for management changes',
                           'optimism about young talent stepping up', 'concern about the team\'s future',
                           'appreciation for the player\'s legacy', 'ticket demand for farewell appearances'],
            'sports_event': ['championship final', 'Grand Slam tournament', 'Olympic showdown',
                           'title fight', 'playoff series', 'historic rivalry match',
                           'season opener', 'international competition', 'all-star game',
                           'world championship', 'charity exhibition', 'invitational tournament'],
            'player1_strength': ['incredible consistency', 'mental toughness', 'technical precision',
                               'explosive speed', 'tactical awareness', 'physical dominance',
                               'clutch performance', 'versatility', 'innovative technique',
                               'endurance', 'strategic brilliance', 'competitive drive'],
            'player1_recent': ['winning three consecutive tournaments', 'setting a new world record',
                             'recovering from a career-threatening injury', 'changing coaches',
                             'introducing a revolutionary technique', 'dominating qualifiers',
                             'experimenting with new equipment', 'focusing on mental preparation',
                             'implementing a transformed training regimen', 'refining their signature move'],
            'player2_strength': ['unorthodox style', 'remarkable agility', 'powerful execution',
                               'analytical approach', 'precision under pressure', 'adaptive strategy',
                               'efficient energy management', 'psychological resilience',
                               'creative problem-solving', 'exceptional focus', 'flawless technique'],
            'sports_commentator': ['John Anderson', 'Martin Tyler', 'Doris Burke', 'Al Michaels',
                                 'Jon Champion', 'Rebecca Lowe', 'Mike Breen', 'Troy Aikman',
                                 'Jessica Mendoza', 'Joe Buck', 'Arlo White', 'Shaquille O\'Neal',
                                 'Maria Taylor', 'Kirk Herbstreit', 'Charles Barkley'],
            'odds': ['slight', 'heavy', '2-to-1', 'overwhelming', 'surprising', 'expected',
                   'narrow', 'considerable', 'strong', 'marginal'],
            'factor': ['recent form', 'home advantage', 'weather conditions', 'tactical matchups',
                     'historical head-to-head records', 'injury recoveries', 'psychological edge',
                     'crowd support', 'rest time between competitions', 'adaptation to the venue'],
            'sports_venue': ['Madison Square Garden', 'Wembley Stadium', 'Staples Center', 'Yankee Stadium',
                           'Camp Nou', 'Roland Garros', 'Augusta National', 'Fenway Park',
                           'AT&T Stadium', 'Old Trafford', 'Arthur Ashe Stadium',
                           'MetLife Stadium', 'TD Garden', 'Dodger Stadium',
                           'National Stadium', 'Olympic Arena', 'Central Sportsplex',
                           'Memorial Coliseum', 'Riverside Park', 'Victory Field'],
            'venue_feature': ['enhanced video boards', 'state-of-the-art surface', 'expanded seating capacity',
                            'upgraded hospitality areas', 'improved sound system', 'additional security measures',
                            'special commemorative displays', 'new camera positions for broadcast',
                            'modified environmental controls', 'streamlined entry procedures'],
            'sports_stake': ['championship implications', 'qualification for international competition',
                           'all-time series record', 'playoff seeding', 'breaking historical ties',
                           'year-end rankings', 'valuable points in the standings',
                           'endorsement opportunities', 'legacy considerations', 'record-breaking potential']
        },
        'health': {
            'health_journal': ['Journal of the American Medical Association', 'The New England Journal of Medicine',
                             'The Lancet', 'British Medical Journal', 'Nature Medicine',
                             'Annals of Internal Medicine', 'PLOS Medicine', 'Mayo Clinic Proceedings',
                             'Science Translational Medicine', 'Journal of Clinical Investigation',
                             'Frontiers in Medicine', 'BMC Medicine'],
            'health_finding': ['a strong association between daily exercise and reduced risk of cardiovascular disease',
                             'significant benefits of intermittent fasting for metabolic health',
                             'promising effects of mindfulness practice on stress reduction',
                             'a potential link between gut microbiome composition and immune function',
                             'compelling evidence supporting early intervention for mental health conditions',
                             'unexpected correlations between sleep quality and cognitive performance'],
            'health_topic': ['heart disease prevention', 'dietary patterns', 'stress management techniques',
                           'microbiome health', 'early detection of mental health issues',
                           'sleep optimization', 'preventative screenings', 'exercise protocols',
                           'cognitive health', 'chronic disease management'],
            'health_institution': ['Harvard Medical School', 'Mayo Clinic', 'Johns Hopkins University',
                                 'Stanford Medicine', 'Cleveland Clinic', 'NIH', 'UC San Francisco',
                                 'Massachusetts General Hospital', 'Karolinska Institute',
                                 'Oxford University Medical Sciences Division', 'UCLA Health',
                                 'University of Michigan Medicine', 'Memorial Sloan Kettering',
                                 'Medical Research Institute', 'Global Health Center',
                                 'National Medical University', 'City Hospital Research Department'],
            'study_type': ['randomized controlled', 'longitudinal', 'cross-sectional', 'prospective cohort',
                         'meta-analysis', 'systematic review', 'case-control', 'observational',
                         'double-blind', 'interventional', 'mixed-methods'],
            'health_result': ['a 45% reduction in risk among the intervention group', 'significant improvements in biomarkers across all demographics',
                            'clinically meaningful changes in patient-reported outcomes',
                            'compelling statistical associations that persisted after adjustment for confounders',
                            'dose-dependent effects that suggest causal relationships',
                            'promising preliminary data that warrants further investigation'],
            'health_application': ['preventative health guidelines', 'clinical practice recommendations',
                                 'lifestyle intervention programs', 'public health initiatives',
                                 'personalized treatment approaches', 'screening protocols',
                                 'patient education strategies', 'healthcare policy development'],
            'health_researcher': ['Jennifer Martinez', 'David Chang', 'Sarah Wilson', 'Mohammed Al-Fayez',
                                'Ravi Patel', 'Elizabeth Taylor', 'Carlos Menendez',
                                'Sophia Kim', 'James Richardson', 'Elena Petrov'],
            'quote_health': ['these findings represent a paradigm shift in how we approach this health challenge',
                           'our research suggests a more nuanced understanding is needed than previously thought',
                           'the data clearly demonstrate that even modest interventions can have substantial impacts',
                           'this study addresses a critical gap in our understanding of these complex mechanisms',
                           'we were surprised by the strength of the association, which exceeded our hypothesis',
                           'these results could fundamentally change clinical guidelines'],
            'health_recommendation': ['engage in at least 150 minutes of moderate exercise weekly',
                                    'incorporate more plant-based foods into their diet',
                                    'practice mindfulness techniques for 10-15 minutes daily',
                                    'prioritize 7-9 hours of quality sleep',
                                    'discuss screening options with their healthcare provider',
                                    'monitor specific biomarkers through regular testing',
                                    'maintain adequate hydration throughout the day',
                                    'limit screen time before bedtime'],
            'health_critic': ['Robert Johnson', 'Maria Gonzalez', 'Peter Williams', 'Susan Chen',
                            'Thomas Wilson', 'Amanda Cooper', 'Richard Thompson',
                            'Julia Martinez', 'Daniel Kim', 'Olivia Robinson'],
            'health_limitation': ['the study was limited by its relatively small sample size',
                                'longer-term follow-up is needed to confirm sustained effects',
                                'these findings may not generalize to all populations',
                                'the observational design limits causal inference',
                                'multiple interventions make it difficult to isolate specific mechanisms',
                                'self-reported data introduces potential reporting bias'],
            'next_steps': ['conduct a larger multicenter trial', 'extend the follow-up period to assess durability of effects',
                         'investigate potential mechanisms through laboratory studies',
                         'explore applications in more diverse populations',
                         'develop and test specific interventions based on these findings',
                         'collaborate with implementation scientists to translate results into practice'],
            'health_question': ['optimal timing and dosage', 'individual variability in response',
                              'potential interactions with existing treatments',
                              'cost-effectiveness in real-world settings',
                              'best delivery methods for different populations',
                              'long-term sustainability of observed benefits'],
            'health_condition': ['type 2 diabetes', 'hypertension', 'anxiety disorders', 'chronic pain',
                               'inflammatory bowel disease', 'osteoporosis', 'migraine',
                               'depression', 'metabolic syndrome', 'asthma', 'autism spectrum disorder',
                               'Alzheimer\'s disease', 'obesity', 'cardiac arrhythmias'],
            'health_demographic': ['children and adolescents', 'older adults', 'pregnant women',
                                 'racial and ethnic minorities', 'rural communities',
                                 'low-income populations', 'healthcare workers',
                                 'individuals with multiple chronic conditions',
                                 'college students', 'remote workers'],
            'health_rate_change': ['seen consistent increases', 'experienced a dramatic decline',
                                 'shown concerning fluctuations', 'stabilized after previous increases',
                                 'demonstrated gradual improvement', 'plateaued at concerning levels',
                                 'varied significantly by geographic region'],
            'health_official': ['Dr. Rebecca Martin', 'Dr. James Chen', 'Dr. Elizabeth Torres',
                              'Dr. Michael Washington', 'Dr. Sophia Patel', 'Dr. David Kim',
                              'Dr. Maria Gonzalez', 'Dr. Jonathan Lee', 'Dr. Tanya Rodriguez',
                              'Dr. William Johnson'],
            'health_organization': ['Centers for Disease Control and Prevention', 'World Health Organization',
                                  'National Institutes of Health', 'American Heart Association',
                                  'Memorial Sloan Kettering Cancer Center', 'American Diabetes Association',
                                  'Mental Health America', 'American Academy of Pediatrics',
                                  'Alzheimer\'s Association', 'American Public Health Association',
                                  'National Patient Alliance', 'Health Advocacy Foundation',
                                  'Community Wellness Coalition', 'Medical Rights Network'],
            'health_factor': ['changes in screening protocols', 'increased public awareness',
                            'expanded access to preventive services', 'environmental factors',
                            'shifts in behavioral patterns', 'implementation of new guidelines',
                            'technological advances in early detection', 'community-based interventions',
                            'policy changes affecting healthcare access', 'improved surveillance systems'],
            'health_prevention': ['community education campaigns', 'school-based health programs',
                                'workplace wellness initiatives', 'telehealth screening services',
                                'mobile health applications', 'targeted outreach to high-risk populations',
                                'integration of preventive services in primary care',
                                'peer support networks', 'public health messaging strategies'],
            'health_outcome': ['reducing disparities in vulnerable communities', 'lowering hospitalization rates',
                             'improving quality of life measures', 'increasing early detection rates',
                             'enhancing treatment adherence', 'facilitating behavior change',
                             'strengthening healthcare engagement', 'building community resilience'],
            'health_initiative': ['mobile health clinics', 'culturally tailored education materials',
                                'community health worker programs', 'intergenerational support groups',
                                'technology-assisted monitoring', 'faith-based outreach programs',
                                'school partnerships', 'workplace screening events',
                                'transportation assistance services'],
            'health_practice': ['regular health screenings', 'vaccination', 'proper hand hygiene',
                              'balanced nutrition', 'adequate physical activity',
                              'stress management techniques', 'sufficient sleep',
                              'substance use reduction', 'social connection maintenance'],
            'health_prediction': ['cases will continue to decline if current interventions are maintained',
                                'a resurgence is possible without sustained preventive efforts',
                                'disparities may widen without targeted interventions',
                                'long-term health system impacts will require strategic planning',
                                'community resilience will be essential for sustainable progress',
                                'evolving variants may present new challenges to control efforts'],
            'health_field': ['immunotherapy', 'gene therapy', 'regenerative medicine', 'precision nutrition',
                           'digital therapeutics', 'microbiome-based treatments', 'neurostimulation',
                           'nanomedicine', 'psychedelic-assisted therapy', 'epigenetic modification',
                           'bioelectronic medicine', 'targeted antibody therapy'],
            'health_treatment': ['novel drug combination', 'minimally invasive procedure',
                               'artificial intelligence-guided therapy', 'personalized vaccine',
                               'gene-editing approach', 'microbiome transplantation',
                               'digital health intervention', 'implantable medical device',
                               'stem cell therapy', 'targeted radiation technique'],
            'treatment_mechanism': ['selectively targeting specific cellular pathways',
                                  'modulating immune response with precision',
                                  'restoring normal tissue function through regeneration',
                                  'correcting genetic mutations at their source',
                                  'recalibrating disrupted neural circuits',
                                  'leveraging the body\'s own healing mechanisms',
                                  'delivering therapeutic agents with unprecedented precision'],
            'health_metric': ['symptom severity', 'quality of life', 'functional capacity',
                            'biomarker levels', 'disease progression', 'treatment adherence',
                            'adverse event frequency', 'hospitalization rates',
                            'relapse prevention', 'cognitive function'],
            'health_side_effect': ['mild gastrointestinal discomfort', 'temporary fatigue',
                                 'short-term sleep disturbances', 'minor injection site reactions',
                                 'transient changes in appetite', 'headaches that typically resolve within 24 hours',
                                 'brief dizziness during initial treatment phases',
                                 'slight alterations in taste perception'],
            'conventional_limitation': ['often become less effective over time',
                                      'frequently cause significant side effects',
                                      'require invasive administration methods',
                                      'fail to address the root cause of the condition',
                                      'work only for a subset of patients',
                                      'necessitate lifelong medication regimens',
                                      'impose substantial lifestyle limitations'],
            'insurance_status': ['is increasingly being covered by major insurers',
                               'remains inconsistently reimbursed across plans',
                               'has received preliminary approval from several payers',
                               'is covered only after failure of conventional options',
                               'requires additional outcomes data for widespread coverage',
                               'is accessible through specialized approval pathways',
                               'varies significantly by geographic region'],
            'health_access_issue': ['geographic disparities in treatment availability',
                                  'high out-of-pocket costs for patients',
                                  'inadequate provider training and certification',
                                  'complex prior authorization requirements',
                                  'limited availability of specialized facilities',
                                  'inequitable distribution of medical expertise',
                                  'socioeconomic barriers to care continuity'],
            'advocacy_goal': ['expand insurance coverage policies', 'establish patient assistance programs',
                            'develop provider education initiatives', 'create regional treatment centers',
                            'streamline approval processes', 'conduct community awareness campaigns',
                            'advocate for health policy reforms', 'secure research funding for implementation studies']
        },
        'entertainment': {
            'entertainment_title': ['The Last Horizon', 'Midnight Chronicles', 'Echoes of Tomorrow',
                                  'Stellar Junction', 'The Hidden Truth', 'Legacy of Shadows',
                                  'Quantum Paradox', 'City of Dreams', 'The Final Cipher',
                                  'Eternal Resonance', 'Whispers in the Dark', 'Neon Dynasty',
                                  'Forgotten Realms', 'The Masterpiece', 'Parallel Lives'],
            'entertainment_director': ['Christopher Nolan', 'Ava DuVernay', 'Denis Villeneuve',
                                     'Greta Gerwig', 'Jordan Peele', 'Chloe Zhao',
                                     'Ryan Coogler', 'Bong Joon-ho', 'Taika Waititi',
                                     'Emerald Chen', 'Marcus Washington', 'Sophia Rodriguez',
                                     'David Park', 'Isabella Martinez', 'Jonathan Lee'],
            'entertainment_star': ['Zendaya', 'Tom Holland', 'Florence Pugh', 'Timothée Chalamet',
                                 'Anya Taylor-Joy', 'Daniel Kaluuya', 'Margot Robbie',
                                 'Michael B. Jordan', 'Lupita Nyong\'o', 'Pedro Pascal',
                                 'Jennifer Lawrence', 'Jonathan Majors', 'Saoirse Ronan',
                                 'Adam Driver', 'Anthony Ramos', 'Ariana Williams',
                                 'Marcus Chen', 'Olivia Taylor', 'Jason Rodriguez'],
            'entertainment_genre': ['science fiction', 'psychological thriller', 'coming-of-age drama',
                                  'historical epic', 'romantic comedy', 'supernatural horror',
                                  'action adventure', 'musical', 'crime noir', 'fantasy',
                                  'biographical drama', 'animated', 'spy thriller',
                                  'mystery', 'mockumentary'],
            'rating': ['universal acclaim', 'mostly positive reviews', 'mixed but generally favorable notices',
                     'critical praise', 'enthusiastic responses', 'overwhelmingly positive feedback',
                     'strong critical endorsement', 'divisive but passionate reactions'],
            'film_element': ['breathtaking cinematography', 'nuanced performances', 'innovative screenplay',
                           'masterful direction', 'haunting score', 'groundbreaking visual effects',
                           'powerful emotional resonance', 'thought-provoking themes',
                           'meticulous production design', 'brilliant ensemble cast'],
            'entertainment_critic': ['Anthony Lane', 'Manohla Dargis', 'David Ehrlich', 'Angelica Jade Bastién',
                                   'Richard Brody', 'Alison Willmore', 'Justin Chang', 'K. Austin Collins',
                                   'Bilge Ebiri', 'Stephanie Zacharek', 'Michael Phillips',
                                   'Dana Stevens', 'Carlos Rivera', 'Emily Taylor', 'Marcus Johnson'],
            'entertainment_outlet': ['The New Yorker', 'The New York Times', 'IndieWire', 'Vulture',
                                   'The Atlantic', 'New York Magazine', 'Los Angeles Times',
                                   'Vanity Fair', 'Variety', 'Time', 'Chicago Tribune',
                                   'Slate', 'The Hollywood Reporter', 'Entertainment Weekly',
                                   'The Guardian', 'Cinema Pulse', 'Screen Journal',
                                   'Film Critics Association', 'The Movie Database', 'Digital Entertainment'],
            'quote_entertainment': ['a stunning achievement that redefines the boundaries of its genre',
                                  'the rare blockbuster that combines spectacle with genuine emotional depth',
                                  'a mesmerizing experience that lingers long after the credits roll',
                                  'showcases a filmmaker at the height of their creative powers',
                                  'features performances of remarkable subtlety and power',
                                  'balances technical virtuosity with authentic human connection'],
            'film_aspect': ['unexpected plot developments', 'complex character arcs', 'immersive world-building',
                          'authentic dialogue', 'innovative narrative structure', 'cultural relevance',
                          'visual storytelling', 'emotional authenticity', 'tonal balance',
                          'thematic cohesion'],
            'entertainment_challenge': ['production delays', 'budget constraints', 'scheduling conflicts',
                                      'technical hurdles', 'pandemic-related restrictions',
                                      'last-minute script revisions', 'difficult shooting conditions',
                                      'post-production complications', 'casting changes'],
            'sequel_status': ['greenlit', 'in early development', 'scheduled for production next year',
                            'confirmed with the original creative team', 'announced at the studio\'s investor day',
                            'being written', 'planned as part of a trilogy', 'fast-tracked by the studio'],
            'music_genre': ['alternative R&B', 'progressive hip-hop', 'experimental pop',
                          'neo-soul', 'electronic', 'indie rock', 'contemporary jazz fusion',
                          'folk-inspired', 'orchestral', 'avant-garde', 'synthwave'],
            'entertainment_collaborator': ['Kendrick Lamar', 'Billie Eilish', 'The Weeknd',
                                         'Taylor Swift', 'Bad Bunny', 'SZA', 'Tyler, the Creator',
                                         'Rosalía', 'Frank Ocean', 'FKA twigs', 'Jorja Smith',
                                         'Anderson .Paak', 'James Blake', 'HAIM', 'Sampha',
                                         'Marcus Lee', 'Skyler Band', 'Elena Rodriguez'],
            'entertainment_producer': ['Jack Antonoff', 'Metro Boomin', 'Finneas', 'Noah "40" Shebib',
                                     'Kenny Beats', 'Sounwave', 'Mike Dean', 'Kaytranada',
                                     'Take a Daytrip', 'Nigel Godrich', 'Linda Perry',
                                     'Mark Ronson', 'Catherine Wang', 'David Thompson',
                                     'Michelle Garcia', 'Rhythm Masters'],
            'music_element': ['innovative sonic textures', 'personal and vulnerable lyrics',
                            'genre-defying composition', 'masterful vocal performance',
                            'seamless production', 'inventive arrangements', 'thematic coherence',
                            'cultural commentary', 'emotional depth', 'technical virtuosity'],
            'music_direction': ['bold new artistic direction', 'return to their roots',
                              'maturation of their sound', 'unexpected stylistic pivot',
                              'refinement of their established aesthetic', 'collaborative reinvention',
                              'conceptual evolution', 'sonic experimentation'],
            'artist_history': ['dominated the charts with their debut', 'struggled with creative differences',
                             'went on a three-year hiatus', 'experimented with different genres',
                             'collaborated with unexpected artists', 'produced their own material',
                             'toured extensively', 'focused on visual storytelling'],
            'song_title': ['Midnight Reflections', 'Celestial Bodies', 'Digital Heartbeat',
                         'Ultraviolet Dreams', 'Phantom Memory', 'Emerald Sky',
                         'Whispers of Forever', 'Chrome Emotions', 'Golden Hour',
                         'Velvet Thunder', 'Crystal Silence', 'Neon Prayer'],
            'chart_position': ['number one', 'the top five', 'the top ten', 'number three',
                             'the top of the streaming charts', 'the Billboard Hot 100',
                             'multiple international charts', 'the highest new entry'],
            'track_name': ['Ethereal Dance', 'Midnight Confession', 'Solar Flare', 'Velvet Revolution',
                         'Quantum Heart', 'Urban Wilderness', 'Electric Soul', 'Diamond Mind',
                         'Astral Projection', 'Holographic', 'Parallel Universe', 'Digital Echo'],
            'track_quality': ['raw emotional vulnerability', 'intricate production', 'mesmerizing vocal performance',
                            'innovative sound design', 'hypnotic beat progression', 'powerful lyrics',
                            'unexpected structural choices', 'seamless genre blending',
                            'memorable melodic hooks', 'atmospheric depth'],
            'entertainment_creator': ['Ryan Murphy', 'Shonda Rhimes', 'Taylor Sheridan',
                                    'Phoebe Waller-Bridge', 'Donald Glover', 'Michaela Coel',
                                    'Dan Fogelman', 'Vince Gilligan', 'Michael Schur',
                                    'Lena Waithe', 'Jesse Armstrong', 'Damon Lindelof',
                                    'Alexandra Chen', 'Marcus Williams', 'Sophia Thompson'],
            'show_element': ['complex character development', 'intricate plot twists', 'innovative narrative structure',
                           'stunning visual aesthetic', 'thought-provoking themes', 'authentic representation',
                           'masterful performances', 'bold storytelling choices', 'immersive world-building',
                           'sophisticated dialogue'],
            'viral_element': ['unexpected plot reveal', 'controversial character decision',
                            'emotional final scene', 'shocking cliffhanger', 'groundbreaking representation',
                            'memorable dialogue exchange', 'visually stunning sequence',
                            'powerful performance moment', 'surprising cameo appearance'],
            'performance_quality': ['nuanced emotional range', 'compelling character transformations',
                                  'authentic chemistry', 'powerful dramatic moments',
                                  'subtle physical characterization', 'commanding screen presence',
                                  'seamless ensemble work', 'breakthrough performances from newcomers',
                                  'masterful delivery of complex dialogue'],
            'streaming_service': ['Netflix', 'HBO Max', 'Disney+', 'Amazon Prime Video',
                                'Apple TV+', 'Hulu', 'Paramount+', 'Peacock',
                                'AMC+', 'Showtime', 'Starz', 'FX on Hulu',
                                'StreamNow', 'VisualPlus', 'MaxView', 'PrimeStream'],
            'production_status': ['currently underway', 'in pre-production', 'scheduled to begin next month',
                                'wrapping up', 'delayed due to scheduling conflicts',
                                'accelerated due to high demand', 'confirmed with the original cast',
                                'being developed with an expanded scope', 'in the final stages of writing'],
            'release_timeframe': ['early next year', 'the summer season', 'the fall lineup',
                                'the holiday window', 'the first quarter', 'the award season',
                                'the streaming platform\'s flagship period', 'a special event release']
        }
    }
    
    # Common word banks for all categories
    common_words = {
        'number': ['25', '50', '100', '150', '200', '500', '1,000', '1,500', '2,000'],
        'number_high': ['50,000', '100,000', '250,000', '500,000', '1 million', '2 million', '5 million'],
        'percent': ['5', '10', '15', '20', '25', '30', '40', '50', '75'],
        'small_percent': ['2', '3', '5', '7', '8', '10', '12'],
        'increase_decrease': ['increase', 'decrease', 'rise', 'decline', 'growth', 'reduction', 'surge', 'drop'],
        'amount': ['$50 million', '$100 million', '$200 million', '$500 million', '$1 billion',
                 '€30 million', '€75 million', '€150 million', '¥5 billion', '£25 million'],
        'amount_total': ['$500 million', '$750 million', '$1 billion', '$1.5 billion',
                        '€300 million', '€500 million', '¥10 billion', '£200 million'],
        'timeframe': ['3 months', '6 months', 'one year', 'two years', 'a decade',
                     'the next quarter', 'the coming year', 'the past year',
                     '24 hours', 'a week', 'the first month', 'the opening weekend'],
        'year': ['2025', '2026', '2027', '2028', '2030'],
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
        'period': ['2024-2025', 'the past fiscal year', 'the previous quarter',
                  'the three-year assessment period', 'January through June',
                  'the initial implementation phase', 'the pilot study duration',
                  'the first six months of operation']
    }
    
    # Generate sample data
    data = []
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    for i in range(n_samples):
        # Select category
        category = random.choice(categories)
        
        # Get template for this category
        templates = templates.get(category, templates['business'])  # Default to business if category not found
        template = random.choice(templates)
        
        # Get word banks for this category
        word_bank = word_banks.get(category, {})
        
        # Fill template with words from category-specific and common word banks
        text = template
        
        # Find all placeholders in the template
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        # Replace each placeholder with a random word from the appropriate word bank
        for placeholder in placeholders:
            if placeholder in word_bank:
                replacement = random.choice(word_bank[placeholder])
            elif placeholder in common_words:
                replacement = random.choice(common_words[placeholder])
            else:
                replacement = f"[{placeholder}]"  # Fallback for missing placeholders
            
            text = text.replace('{' + placeholder + '}', replacement)
        
        # Determine text length
        min_length, max_length = lengths
        target_length = random.randint(min_length, max_length)
        
        # Adjust text length if needed
        current_length = len(text.split())
        
        if current_length < target_length:
            # Text is too short, add more sentences
            additional_sentences_needed = (target_length - current_length) // 10 + 1
            
            for _ in range(additional_sentences_needed):
                if category in templates:
                    # Get an additional short template
                    add_template = random.choice(templates[category])
                    
                    # Fill placeholders
                    add_text = add_template
                    add_placeholders = re.findall(r'\{([^}]+)\}', add_template)
                    
                    for placeholder in add_placeholders:
                        if placeholder in word_bank:
                            replacement = random.choice(word_bank[placeholder])
                        elif placeholder in common_words:
                            replacement = random.choice(common_words[placeholder])
                        else:
                            replacement = f"[{placeholder}]"
                        
                        add_text = add_text.replace('{' + placeholder + '}', replacement)
                    
                    # Add the sentence
                    text = text + " " + add_text
        
        elif current_length > max_length:
            # Text is too long, truncate it
            words = text.split()
            text = ' '.join(words[:max_length]) + '.'
        
        # Add sample to dataset
        data.append({
            'text': text,
            'category': category,
            'length': len(text.split())
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output file provided
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Generated {n_samples} synthetic text samples saved to '{output_file}'")
    
    return df

def generate_test_environment():
    """
    Generate a complete test environment with synthetic emails, documents, and text data
    """
    # Create data directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/emails', exist_ok=True)
    os.makedirs('data/documents', exist_ok=True)
    os.makedirs('data/text_classification', exist_ok=True)
    
    # Generate synthetic emails
    print("Generating synthetic emails...")
    emails_df = generate_synthetic_emails(
        n_emails=50,
        output_folder='data/emails',
        date_range=('2024-01-01', '2025-05-01'),
        save_files=True
    )
    
    # Generate synthetic documents
    print("\nGenerating synthetic documents...")
    documents_df = generate_synthetic_documents(
        n_documents=30,
        output_folder='data/documents',
        date_range=('2024-01-01', '2025-05-01'),
        save_files=True
    )
    
    # Generate synthetic text classification dataset
    print("\nGenerating synthetic text classification dataset...")
    text_df = generate_synthetic_text_dataset(
        n_samples=200,
        output_file='data/text_classification/text_samples.csv'
    )
    
    print("\nTest environment generation complete!")
    return {
        'emails': emails_df,
        'documents': documents_df,
        'text_classification': text_df
    }