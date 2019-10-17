from analyse.doc2ve_clustering import main_process
from analyse.get_profiles import  get_full_profile
from access_search_db import connect_and_insert
from datetime import date
from dateutil.relativedelta import relativedelta    



def scoring_profile(desired_profile):
    new_job={"title":desired_profile["title"], "summary":desired_profile["summary"]}
    profiles = main_process(new_job)
    full_profiles = []
    for profile in profiles:
        p = get_full_profile(str(profile))
        p["_id"] = (str(p["_id"]))
        full_profiles.append(p)
    scores_totals = []
    for profile in full_profiles:
        scor_ex = 0 
        score_ev = 0
        score_skills = 0 
        desired_skills = desired_profile["skills"]
        skills = profile["skills"]
        unit = 50/len(desired_skills)
        for ds in desired_skills:
            for s in skills:
                if ds.lower() == s.lower():
                    score_skills += unit
        ev = evolution(p)
        if ev:
            score_ev = 20
        experience = count_years_of_experience(p)
        if experience !=1 :
            if experience <= 5:
                scor_ex += 5
            if experience <= 10 and experience >5:
                scor_ex += 10
            if experience <= 15 and experience >10:
                scor_ex += 15
            if experience >15:
                scor_ex += 20
        score_total = score_skills+score_ev+scor_ex
        profile_score={"profile_id":profile['_id'], "score":score_total}
        profile["score"] = int(score_total)
        scores_totals.append(int(score_total))

    full_profiles = sorted(full_profiles, key=lambda k: k["score"], reverse=True)
    scores_totals = sorted(scores_totals, reverse=True)
    connect_and_insert(desired_profile, scores_totals)
    return full_profiles

def evolution(profile):
    evolution = False
    positions_titles = []
    positions = profile["positions"]
    hierarchy = []
    for position in positions:
        positions_titles.append(position["title"])
    for pt in positions_titles:
        h = detrmine_hierarchy(pt)
        if h != -1 :
            hierarchy.append(h)
    if(len(hierarchy) != 0):
        for i in range(len(hierarchy)-1):
            if (hierarchy[i] != hierarchy[i+1]):
                evolution = True
                return evolution

def detrmine_hierarchy(title):
    entry = ["entry", "intermediate", "junior", "jr.", "jr"]
    support = [ "senior", "highly skilled", "specialist"]
    management_and_professional = ["developing", "career", "advanced", "expert", "principal"]
    business_leadership = ["supervisor", "sr. supervisor", "manager", "sr. manager", "director", "sr. director", "vice president","vp", "Chief", "ceo", "cfo", "coo", "cto"]
    for w in title:
        if w in entry:
            return 0
        if w in support:
            return 1 
        if w in management_and_professional:
            return 2 
        if w in business_leadership:
            return 3 
        return -1


def count_years_of_experience(profile):
    try:
        positions = profile["positions"]
        dates = []
        for p in positions:
            
            s = p["start_date"]
            sd = date(*map(int, s.split('-')))
            try:
                e = p["end_date"]
                ed = date(*map(int, e.split('-')))
            except:
                e = "current"
                ed = date.today()
            dates.append(sd)
            dates.append(ed)
        dates = sorted(dates)
        diff = relativedelta(dates[len(dates)-1], dates[0]).years
        return diff
    except:
        return -1

    

def main():
    desired_profile = {"title":"Sr. Java Developper" ,"summary":"We are looking for a Java Developer with experience in building high-performing, scalable, enterprise-grade applications.You will be part of a talented software team that works on mission-critical applications. Java developer roles and responsibilities include managing Java/Java EE application development while providing expertise in the full software development lifecycle, from concept and design to testing.", "skills":["java", "j2EE"]}
    """p ,s = scoring_profile(desired_profile)
    print(s)"""
    new_job={"title":desired_profile["title"], "summary":desired_profile["summary"]}
    profiles = main_process(new_job)
    full_profiles = []
    for profile in profiles:
        p = get_full_profile(str(profile))
        p["_id"] = (str(p["_id"]))
        full_profiles.append(p)
    for profile in full_profiles:
        print(profile['_id'])
        exp = count_years_of_experience(profile)
        

"""
if __name__== "__main__":
  main()"""