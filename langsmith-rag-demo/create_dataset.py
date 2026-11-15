from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

Dataset_Name = "hr-policy-rag-eval-data"

qa_pairs = [
    ("How many earned leaves is a regular employee entitled to in a calendar year?",
     "18 earned leaves per calendar year. If the employee joins mid-year, earned leaves are granted on a pro rata basis."),
    ("If an employee joins on the 8th of the month, how many leaves are credited?",
     "1.5 leaves, since the employee joined before or on the 10th of the month."),
    ("What leave credit applies to an employee joining between the 11th and 18th of a month?",
     "1 leave."),
    ("Does an employee joining after the 20th of the month get any leave credit?",
     "No credit is given for joining after the 20th of the month."),
    ("Is any leave given on an employee's birthday?",
     "Yes, 0.5 leave is given to the employee on their birthday."),
    ("Can an employee take paid leave during the first three months?",
     "No. For the initial three months no paid leave is allowed."),
    ("Is leave allowed during the notice period?",
     "No leave is allowed during the notice period."),
    ("What is the maximum casual leave a staff member can avail?",
     "A maximum of 12 days of casual leave in a calendar year, normally subject to a maximum of 3 days at a time. The 3-day limit may be relaxed in special circumstances at the discretion of the HR Department."),
    ("What happens if an employee is absent for more than 5 days without informing anyone?",
     "If a staff member remains absent from duty for more than 5 days without intimation to the concerned authorities, their contract is liable to be terminated by the concerned authority."),
    ("How many leaves can be carried forward to the next year?",
     "A maximum of 6 leaves can be carried forward. Remaining leaves at the end of the year lapse automatically."),
    ("What is the maternity leave entitlement?",
     "A female employee with over one year of continuous service is entitled to 26 weeks of maternity leave on full pay, subject to a medical certificate confirming pregnancy and the probable date of confinement. Other leave may be combined provided the total does not exceed 60 days. For employees covered under ESIC, maternity leave is provided by ESIC."),
    ("What are the working days and office timings?",
     "Monday to Saturday, from 9:30 am to 6:00 pm."),
    ("What is the weekly off policy?",
     "Sunday is the weekly off, and the 2nd Saturday of every month is off. Weekly offs on Sunday may be cancelled by management at its discretion depending on workload."),
    ("What are the lunch break rules?",
     "One hour of lunch break per day, which can be taken anytime between 1:00 pm and 2:00 pm on a staggered schedule so that an employee's absence does not create a problem for colleagues."),
    ("How is attendance recorded for field employees and branches?",
     "Attendance for field employees and branches is recorded through the Lystloc mobile tracking app."),
    ("What is the flexi-entry rule for late arrivals?",
     "Employees must arrive by 9:30 am. A flexi-entry is allowed only two times in a month, with the intention of clocking 9 hours for the day. Failing this is treated as absence and deemed a half day. Repeated default is considered indiscipline and can lead to strict disciplinary action."),
    ("Who should a whistleblower report unethical activity to?",
     "The matter should first be shared confidentially with the reporting manager, then with the HR Department, and if unresolved, with the Vigilance Officer."),
    ("Who is the HR Manager and what are their contact details?",
     "Ankit Pradhan, HR Manager. Email: Hr@rikalp.in, Contact Number: 8696202041. Calls are taken between 09:30 am and 6:00 pm on working days."),
    ("What are the personal car and bike travel allowance rates?",
     "Personal car allowance is INR 6 per KM for petrol or diesel, applicable from Slab 07 onwards. Bike allowance is INR 3 per KM, applicable for Slabs 01 to 06."),
    ("What is the retirement age at Rikalp Capital?",
     "The HR policy document does not specify this."),
]

if client.has_dataset(dataset_name=Dataset_Name):
    client.delete_dataset(dataset_name=Dataset_Name)

dataset = client.create_dataset(
    dataset_name=Dataset_Name,
    description="QA pairs from the Rikalp Capital HR Policy PDF for RAG evaluation."
)

client.create_examples(
    dataset_id=dataset.id,
    examples=[{"inputs": {"question": q}, "outputs": {"answer": a}}
              for q, a in qa_pairs]
)

print(f"Created dataset {Dataset_Name} with {len(qa_pairs)} examples.")
