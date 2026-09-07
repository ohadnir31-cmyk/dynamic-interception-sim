# תיקון ספירת יירוטים מאוחרים - קובצי Git והוראות הרצה

## מה מחליפים ומה מוסיפים

מחליפים רק `src/sim/env.py`. קובץ זה כולל את אותו תיקון ממוקד שנמסר בשיחה.
הקובץ הוכן מקוד המקור שנקרא בגרסה `2a4f1df6449443fd6f0ade8d97b11fb65876bf7c`.
מקור: https://github.com/ohadnir31-cmyk/dynamic-interception-sim.git

מוסיפים את התיקייה `tools/boundary_fix` לשורש המאגר:
- `apply_boundary_fix.py`: מחיל את התיקון על המקור. אין להריץ שוב אחרי החלפה בקובץ המתוקן.
- `test_real_checkout.py`: בדיקות מגע לפני החצייה, אחריה ובדיוק בזמן החצייה.
- `replay_saved_scenarios.py`: הרצה מתוך פרמטרי תרחישים שמורים; לא מחולל חלופי.
- `run_paired_check.py`: פקודת הרצה אחת לבדיקות ולהשוואת שתי גרסאות.
- `boundary_capture.diff`: תיעוד שינוי הקוד.
- קובץ ההוראות הזה ודוח הבדיקות המקומיות.

לא מעדכנים heuristics.py, runner.py, rollout_labeling.py או את קובצי האימון.
לא מעלים את audit_core.py מהחבילה האבחונית, נתוני ניסוי, גיבויים או קובצי pyc.
החבילה אינה מכילה את כל המאגר ואינה מחליפה אותו. היא מכילה רק את הקבצים המצוינים.

## עדכון באמצעות Git במחשב

תחילה, מתוך עותק מאגר נקי, יוצרים ענף המבוסס על גרסת ההצעה. אין להפעיל פקודות reset או clean:

```bash
git status --short
git switch -c boundary-capture-fix 2a4f1df6449443fd6f0ade8d97b11fb65876bf7c
```

אם יש שינויים מקומיים, יש לשמור אותם לפני החלפת הקובץ; אין לדרוס אותם.
עתה מעתיקים את src/sim/env.py ואת tools/boundary_fix מתוך החבילה לאותם נתיבים במאגר.
אין למחוק את התיקיות src או tools הקיימות, ואין להחליפן כתיקיות שלמות.

```bash
python -m pip install -r requirements.txt pytest
python tools/boundary_fix/test_real_checkout.py --repo . --version fixed
python -m pytest -q tests
git diff -- src/sim/env.py
```

רק לאחר שהבדיקות הצליחו:

```bash
git add src/sim/env.py tools/boundary_fix/*.py tools/boundary_fix/*.md tools/boundary_fix/*.diff tools/boundary_fix/*.json
git commit -m "Fix late boundary capture scoring and add paired checks"
git push -u origin boundary-capture-fix
```

הפקודות הן הוראות לביצוע אצלך; החבילה אינה עושה commit או push בעצמה.
ניתן גם לעדכן קבצים דרך אתר GitHub בענף נפרד, תוך הקפדה על הנתיבים המלאים.
על ענף שבוצעו בו שינויים נוספים מאז גרסת ההצעה יש למזג את התיקון בזהירות ולא להחליף קוד בעיוורון.

## הרצה ראשונה

מתוך שורש המאגר המתוקן (התיקייה הכוללת src ו-requirements.txt):

```bash
python tools/boundary_fix/run_paired_check.py --repo . --n-scenarios 20 --seed 20260906 --output-dir outputs/boundary_pair_20_run1
```

בחר תיקיית פלט חדשה בכל הרצה. זו בדיקה חדשה, לא שחזור של הטבלאות.
הסקריפט משווה את הקוד מול גרסת ההצעה לפני תחילת הניסוי, יוצר שני עותקי מקור זמניים,
מריץ בדיקות בשניהם ומריץ חמש יוריסטיקות על אותם פרמטרים שמורים.
קוד הבסיס נלקח מ-git archive ולא מסימולטור שנכתב מחדש.
הסקריפט אינו משנה את מאגר העבודה או את הענף הנוכחי ואינו מאמן מודל.
אם יש שינוי נוסף בתוך src או requirements.txt, הוא עוצר כדי לא לערבב גורמים.

## תוצאות

בקובץ boundary_fix_comparison.csv:
- original_mean: ממוצע מספר היירוטים המקורי.
- fixed_mean: ממוצע לאחר התיקון.
- removed_mean: מספר היירוטים שנגרעו בממוצע לתרחיש, לא אחוזים.
- removed_total: סך היירוטים שנגרעו.
- affected_scenarios: מספר התרחישים שבהם הציון השתנה.

נשמרים גם paired_results.csv, מקור ומקביל מתוקן תחת original/ ו-fixed/,
לוגים, מזהי commit, העתקי env.py וגרסאות הספריות.
NI הוא שם הקוד של NT במסמך.

## שחזור תרחישים ישנים

החלף OLD_RESULTS בתיקייה שבה נשמרו קובצי הניסוי המקורי:

```bash
python tools/boundary_fix/run_paired_check.py --repo . --params-csv "OLD_RESULTS/large_scale_scenario_params.csv" --original-results "OLD_RESULTS/large_scale_full_heuristic_rollouts.csv" --limit 100 --output-dir outputs/boundary_original_100_run1
```

הסקריפט דורש התאמה מדויקת בין הרצת הקוד המקורי לבין ציוני העבר לפני שהוא מריץ את המתוקן.
כדי להריץ את כל התרחישים השמורים, מחליפים --limit 100 ב-`--limit 0` ומשתמשים בתיקיית פלט חדשה.
אין צורך לנחש את ה-seed הראשי. אין לאלץ מעבר כאשר השחזור המקורי אינו תואם.
שחזור זה עוסק בחמש היוריסטיקות הקבועות על תרחישים מלאים בלבד: לא באימון,
לא בטבלאות מצבי ההמשך, לא בבורר המאומן ולא בדוגמאות האורקל.

## Colab

המחברת Run_boundary_fix_from_GitHub.ipynb מורידה כברירת מחדל את הענף boundary-capture-fix.
יש קודם להעלות את הקבצים לענף זה. אין במחברת החלה חוזרת של התיקון ואין checkout של גרסת ההצעה.
גרסת ההצעה משמשת רק עותק ביקורת פנימי בתוך כלי ההשוואה.
התוצאות נשמרות ב-Drive בתיקייה חדשה. הרצה בסביבה חדשה היא הדרך הפשוטה להימנע ממחלקות ישנות בזיכרון.

## גבולות האימות

בסביבת ההכנה לא ניתן היה לבצע git clone של המאגר. קובץ env.py הועתק מתוך קוד המקור
הפומבי בגרסת ההצעה ולא מתוך חבילת audit_core. עליו הורץ התיקון המקורי והורצו הבדיקות האנליטיות.
נבדקו גם תחביר, ממשק הפקודות, שמירת גיבוי וסירוב להחלה כפולה,
בדיקת התאמה למקור Git על מאגר מקומי קטן וחשבונות השוואה על נתוני בדיקה מלאכותיים.
לא בוצעה הרצת כל המאגר, מודלי המשתמש או התרחישים המקוריים בסביבת ההכנה.
לפני הרצה אצלך, run_paired_check.py משווה את env.py המתוקן לתיקון על git show של המקור האמיתי,
ובודק שאין שינוי סמנטי נוסף בקובץ; לכן אי-התאמה תעצור ולא תהפוך לתוצאה מדומה.

זהו תיקון צר לספירה בסוף צעד, לא סימולטור רציף מלא ולא תיקון מסנן TTI או Slack.
