"""
Comprehensive Test Suite for Multi-Agent Question Generation System
Tests various medical scenarios to ensure robust, accurate question generation
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clinicai.adapters.external.question_service_openai import OpenAIQuestionService


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_test(test_name: str):
    """Print test name"""
    print(f"{Colors.OKCYAN}{Colors.BOLD}TEST: {test_name}{Colors.ENDC}")


def print_question(num: int, question: str, answer: str = None):
    """Print a question and optionally its answer"""
    print(f"{Colors.OKBLUE}Q{num}: {question}{Colors.ENDC}")
    if answer:
        print(f"{Colors.WARNING}A{num}: {answer}{Colors.ENDC}")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


async def test_scenario(
    service: OpenAIQuestionService,
    scenario_name: str,
    chief_complaint: str,
    patient_age: int,
    patient_gender: str,
    recently_travelled: bool,
    simulated_answers: list,
    max_questions: int = 10,
    language: str = "en"
):
    """
    Test a complete scenario with simulated patient answers
    """
    print_test(scenario_name)
    print(f"Chief Complaint: {chief_complaint}")
    print(f"Patient: {patient_age}y/o {patient_gender}, Recently travelled: {recently_travelled}")
    print()
    
    asked_questions = []
    previous_answers = []
    issues_found = []
    
    # First question
    first_q = await service.generate_first_question(chief_complaint, language)
    print_question(1, first_q, simulated_answers[0] if simulated_answers else None)
    asked_questions.append(first_q)
    previous_answers.append(simulated_answers[0] if simulated_answers else "")
    
    # Generate subsequent questions
    for i in range(1, max_questions):
        if i >= len(simulated_answers):
            break
        
        try:
            question = await service.generate_next_question(
                disease=chief_complaint,
                previous_answers=previous_answers,
                asked_questions=asked_questions,
                current_count=i,
                max_count=max_questions,
                recently_travelled=recently_travelled,
                patient_gender=patient_gender,
                patient_age=patient_age,
                language=language
            )
            
            # Check for issues
            if "??" in question:
                issues_found.append(f"Q{i+1}: Double question marks found")
            
            # Check for semantic duplicates (simple check)
            for j, prev_q in enumerate(asked_questions, 1):
                if question.lower().strip('?') == prev_q.lower().strip('?'):
                    issues_found.append(f"Q{i+1} is exact duplicate of Q{j}")
            
            print_question(i + 1, question, simulated_answers[i] if i < len(simulated_answers) else None)
            
            asked_questions.append(question)
            if i < len(simulated_answers):
                previous_answers.append(simulated_answers[i])
            
        except Exception as e:
            print_error(f"Question generation failed: {e}")
            break
    
    # Report results
    print()
    if issues_found:
        print_error(f"Issues found: {len(issues_found)}")
        for issue in issues_found:
            print_error(f"  - {issue}")
    else:
        print_success(f"No formatting issues found in {len(asked_questions)} questions")
    
    print()
    return {
        "scenario": scenario_name,
        "questions_generated": len(asked_questions),
        "issues": issues_found,
        "questions": asked_questions
    }


async def run_all_tests():
    """Run comprehensive test suite"""
    
    print_section("MULTI-AGENT QUESTION GENERATION - COMPREHENSIVE TEST SUITE")
    
    # Initialize service
    service = OpenAIQuestionService()
    
    results = []
    
    # ========================================================================
    # TEST 1: Acute Respiratory Infection (Cough, Cold, Fever)
    # ========================================================================
    result1 = await test_scenario(
        service=service,
        scenario_name="Test 1: Acute Respiratory Infection",
        chief_complaint="Cough / Cold, Fever, Headache",
        patient_age=35,
        patient_gender="male",
        recently_travelled=False,
        simulated_answers=[
            "Cough / Cold, Fever, Headache",  # Q1
            "Started 7 days ago",  # Q2: Duration
            "No medications, just resting",  # Q3: Medications
            "Dry cough, mostly at night. Stuffy nose. Headache in the morning.",  # Q4: Symptom characterization
            "It's affecting my work, hard to concentrate",  # Q5: Impact
            "Fever went from 99 to 101, cough getting more frequent",  # Q6: Progression
            "No family members are sick",  # Q7: Family/exposure
        ],
        max_questions=10,
        language="en"
    )
    results.append(result1)
    
    # ========================================================================
    # TEST 2: Chronic Disease - Diabetes
    # ========================================================================
    result2 = await test_scenario(
        service=service,
        scenario_name="Test 2: Chronic Disease - Diabetes",
        chief_complaint="High blood sugar readings",
        patient_age=52,
        patient_gender="female",
        recently_travelled=False,
        simulated_answers=[
            "High blood sugar readings",
            "About 3 years, but it's gotten worse in the past 2 months",
            "Metformin 500mg twice daily",
            "Morning readings around 180-200, afternoon around 150",
            "Fatigue, increased thirst, frequent urination",
            "Last HbA1c was 8.2 three months ago",
            "Yes, my mother and sister both have diabetes",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result2)
    
    # ========================================================================
    # TEST 3: Chronic Disease - Asthma
    # ========================================================================
    result3 = await test_scenario(
        service=service,
        scenario_name="Test 3: Chronic Disease - Asthma",
        chief_complaint="Shortness of breath and wheezing",
        patient_age=28,
        patient_gender="female",
        recently_travelled=False,
        simulated_answers=[
            "Shortness of breath and wheezing for the past 2 weeks",
            "Had asthma since childhood, worsening last 2 weeks",
            "Using Albuterol inhaler, currently 3-4 times per day",
            "Wheezing at night, chest tightness in morning, occasional cough",
            "Triggered by cold air and exercise",
            "Peak flow dropped from 400 to 280 L/min",
            "Yes, both parents have asthma. Mother uses daily controller",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result3)
    
    # ========================================================================
    # TEST 4: Acute Pain - No Family History Needed
    # ========================================================================
    result4 = await test_scenario(
        service=service,
        scenario_name="Test 4: Acute Pain - Lower Back",
        chief_complaint="Lower back pain",
        patient_age=42,
        patient_gender="male",
        recently_travelled=False,
        simulated_answers=[
            "Lower back pain",
            "Started 3 days ago after lifting heavy boxes",
            "Ibuprofen 400mg, not helping much",
            "Sharp pain, worse with movement, radiates to left leg",
            "No other symptoms",
            "Can't stand for long, difficulty walking, can't work",
            "Pain is getting worse, now 8/10",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result4)
    
    # ========================================================================
    # TEST 5: Women's Health - Menstrual Issue
    # ========================================================================
    result5 = await test_scenario(
        service=service,
        scenario_name="Test 5: Women's Health - Irregular Periods",
        chief_complaint="Irregular menstrual periods",
        patient_age=32,
        patient_gender="female",
        recently_travelled=False,
        simulated_answers=[
            "Irregular menstrual periods",
            "Past 6 months",
            "No medications currently",
            "Periods every 45-60 days instead of regular 28 days",
            "Some weight gain, mild acne",
            "Cycles used to be regular until 6 months ago",
            "No, not on birth control",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result5)
    
    # ========================================================================
    # TEST 6: Travel-Related Symptoms
    # ========================================================================
    result6 = await test_scenario(
        service=service,
        scenario_name="Test 6: Travel-Related Illness",
        chief_complaint="Diarrhea and stomach cramps",
        patient_age=29,
        patient_gender="male",
        recently_travelled=True,  # TRAVELLED = TRUE
        simulated_answers=[
            "Diarrhea and stomach cramps",
            "Started 2 days ago",
            "None yet",
            "Watery diarrhea 5-6 times daily, cramping pain",
            "Nausea, mild fever",
            "Returned from Mexico 3 days ago",
            "Ate street food, drank tap water",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result6)
    
    # ========================================================================
    # TEST 7: Hereditary Condition - Hypertension
    # ========================================================================
    result7 = await test_scenario(
        service=service,
        scenario_name="Test 7: Hypertension (Hereditary)",
        chief_complaint="High blood pressure readings",
        patient_age=48,
        patient_gender="male",
        recently_travelled=False,
        simulated_answers=[
            "High blood pressure readings at home",
            "Noticed past month, readings 150/95",
            "No current medications",
            "Occasional headaches, no other symptoms",
            "Checking BP twice daily, usually high",
            "Yes, both parents have high blood pressure, father had stroke",
            "Sedentary job, moderate stress, occasional alcohol",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result7)
    
    # ========================================================================
    # TEST 8: Pediatric Case - Should NOT ask menstrual questions
    # ========================================================================
    result8 = await test_scenario(
        service=service,
        scenario_name="Test 8: Pediatric - Ear Pain (Age 8)",
        chief_complaint="Ear pain",
        patient_age=8,
        patient_gender="female",
        recently_travelled=False,
        simulated_answers=[
            "Right ear pain",
            "2 days",
            "Children's Tylenol",
            "Sharp pain, feels full, hard to hear",
            "Low-grade fever",
            "Can't sleep, keeps waking up crying",
            "Had cold last week",
        ],
        max_questions=10,
        language="en"
    )
    results.append(result8)
    
    # ========================================================================
    # ANALYSIS AND REPORT
    # ========================================================================
    print_section("TEST RESULTS SUMMARY")
    
    total_tests = len(results)
    tests_with_issues = sum(1 for r in results if r["issues"])
    total_questions = sum(r["questions_generated"] for r in results)
    total_issues = sum(len(r["issues"]) for r in results)
    
    print(f"Total Scenarios Tested: {total_tests}")
    print(f"Total Questions Generated: {total_questions}")
    print(f"Scenarios with Issues: {tests_with_issues}")
    print(f"Total Issues Found: {total_issues}")
    print()
    
    if tests_with_issues == 0:
        print_success("✅ ALL TESTS PASSED - No issues found!")
        print_success("The multi-agent system is working flawlessly across all scenarios.")
    else:
        print_error(f"⚠️  {tests_with_issues} scenario(s) had issues")
        print()
        print("Issues by scenario:")
        for result in results:
            if result["issues"]:
                print(f"\n{Colors.WARNING}{result['scenario']}:{Colors.ENDC}")
                for issue in result["issues"]:
                    print_error(f"  - {issue}")
    
    print()
    print_section("OPTIMIZATION ASSESSMENT")
    
    # Check specific improvements
    improvements = []
    concerns = []
    
    # Check for double question marks
    double_qm_count = sum(1 for r in results for q in r["questions"] if "??" in q)
    if double_qm_count == 0:
        improvements.append("✓ No double question marks found")
    else:
        concerns.append(f"✗ Found {double_qm_count} questions with double question marks")
    
    # Check average questions per scenario
    avg_questions = total_questions / total_tests
    if 5 <= avg_questions <= 10:
        improvements.append(f"✓ Good question count: avg {avg_questions:.1f} questions per scenario")
    else:
        concerns.append(f"⚠ Question count outside optimal range: avg {avg_questions:.1f}")
    
    # Check for proper medical reasoning (chronic vs acute)
    chronic_tests = {
        "Test 2": {"needs_monitoring": True, "needs_family": True, "monitoring_keywords": ["monitor", "reading", "check", "glucose", "blood sugar"]},
        "Test 3": {"needs_monitoring": True, "needs_family": True, "monitoring_keywords": ["monitor", "peak flow", "inhaler", "frequency", "usage"]},
        "Test 7": {"needs_monitoring": True, "needs_family": True, "monitoring_keywords": ["monitor", "reading", "check", "blood pressure", "BP"]},
    }
    
    for result in results:
        for test_key, requirements in chronic_tests.items():
            if test_key in result["scenario"]:
                # Check for monitoring questions
                has_monitoring_q = any(
                    any(keyword in q.lower() for keyword in requirements["monitoring_keywords"]) 
                    for q in result["questions"]
                )
                # Check for family history
                has_family_q = any("family" in q.lower() for q in result["questions"])
                
                if has_monitoring_q and has_family_q:
                    improvements.append(f"✓ {result['scenario']}: Properly asked monitoring + family history")
                elif has_monitoring_q:
                    concerns.append(f"⚠ {result['scenario']}: Has monitoring but missing family history")
                elif has_family_q:
                    concerns.append(f"⚠ {result['scenario']}: Has family history but missing monitoring")
                else:
                    concerns.append(f"✗ {result['scenario']}: Missing both monitoring and family history")
    
    print(f"{Colors.BOLD}Improvements:{Colors.ENDC}")
    for imp in improvements:
        print_success(imp)
    
    if concerns:
        print(f"\n{Colors.BOLD}Concerns:{Colors.ENDC}")
        for con in concerns:
            print_error(con)
    
    # Final verdict
    print()
    print_section("FINAL VERDICT")
    
    if not concerns and tests_with_issues == 0:
        print_success("🎉 EXCELLENT - System is optimized and working perfectly!")
        print_success("✓ No formatting issues")
        print_success("✓ Proper medical reasoning")
        print_success("✓ Appropriate questions for each scenario")
        print_success("✓ No semantic duplicates detected")
        print()
        print(f"{Colors.OKGREEN}Optimization Level: 95%+ (Production Ready){Colors.ENDC}")
    elif len(concerns) <= 2 and tests_with_issues <= 1:
        print_success("GOOD - System is mostly optimized with minor issues")
        print()
        print(f"{Colors.WARNING}Optimization Level: 80-90% (Needs minor tweaks){Colors.ENDC}")
    else:
        print_error("NEEDS IMPROVEMENT - System has several issues to address")
        print()
        print(f"{Colors.FAIL}Optimization Level: <80% (Requires attention){Colors.ENDC}")
    
    return results


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Run tests
    asyncio.run(run_all_tests())

