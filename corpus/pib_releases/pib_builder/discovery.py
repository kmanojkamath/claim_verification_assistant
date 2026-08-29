"""PIB release-date selection and release-link discovery."""

import re
from datetime import datetime
from urllib.parse import urljoin

from .config import PIB_BASE
from .text_utils import normalize_text

async def select_date(page, date_string):
    """
    Select a date on the PIB All Releases page.

    PIB has changed its frontend structure over time. Do not assume that
    date controls are always page.locator("select").nth(3/4/5). We inspect
    the main page and every frame and identify the date controls by their
    available options.
    """

    target = datetime.strptime(date_string, "%d-%m-%Y")

    day = str(target.day)
    month = target.strftime("%B")
    year = str(target.year)

    print("\nSelecting:")
    print("Day   :", day)
    print("Month :", month)
    print("Year  :", year)

    # --------------------------------------------------------
    # Give the page JavaScript time to finish rendering.
    # --------------------------------------------------------

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=30000)
    except Exception:
        pass

    await page.wait_for_timeout(2500)

    # --------------------------------------------------------
    # Find the frame containing the date controls.
    # --------------------------------------------------------

    async def find_date_frame():
        frames = page.frames

        print(f"\nInspecting {len(frames)} page/frame(s) for date controls...")

        for frame_number, frame in enumerate(frames):
            try:
                selects = frame.locator("select")
                count = await selects.count()

                print(
                    f"  Frame {frame_number}: "
                    f"{count} <select> element(s)"
                )

                if count >= 3:
                    return frame, selects

            except Exception:
                continue

        return None, None

    frame, selects = await find_date_frame()

    # --------------------------------------------------------
    # Standard PIB structure.
    #
    # Historically:
    #   0 = office
    #   1 = language
    #   2 = ministry
    #   3 = day
    #   4 = month
    #   5 = year
    #
    # However, we now identify day/month/year by their option values
    # instead of blindly relying on fixed positions.
    # --------------------------------------------------------

    if frame is not None:

        count = await selects.count()

        candidates = []

        for i in range(count):
            try:
                options = await selects.nth(i).locator(
                    "option"
                ).all_text_contents()

                options = [
                    normalize_text(x)
                    for x in options
                    if normalize_text(x)
                ]

                candidates.append((i, options))

            except Exception:
                continue

        # ----------------------------------------------------
        # Find YEAR select
        # ----------------------------------------------------

        year_index = None

        for i, options in candidates:
            if year in options:
                year_index = i
                break

        # ----------------------------------------------------
        # Find MONTH select
        # ----------------------------------------------------

        month_index = None

        for i, options in candidates:
            lowered = {x.lower() for x in options}

            if month.lower() in lowered:
                month_index = i
                break

        # ----------------------------------------------------
        # Find DAY select
        # ----------------------------------------------------

        day_index = None

        for i, options in candidates:
            if day in options:
                # Avoid choosing year/month selectors that happen
                # to contain numeric values.
                if i == year_index or i == month_index:
                    continue

                numeric_values = []
                for value in options:
                    if value.isdigit():
                        numeric_values.append(int(value))

                if numeric_values and max(numeric_values) <= 31:
                    day_index = i
                    break

        print(
            f"Detected date controls: "
            f"day={day_index}, "
            f"month={month_index}, "
            f"year={year_index}"
        )

        if (
            day_index is not None
            and month_index is not None
            and year_index is not None
        ):

            year_select = selects.nth(year_index)

            print("Selecting year...")
            await year_select.select_option(label=year)

            # PIB may perform an ASP.NET postback.
            await page.wait_for_timeout(2000)

            # Re-discover the controls after postback.
            frame, selects = await find_date_frame()

            if frame is None:
                print(
                    "❌ Date controls disappeared after year selection."
                )
                return False

            # Re-detect indices.
            candidates = []

            for i in range(await selects.count()):
                try:
                    options = await selects.nth(i).locator(
                        "option"
                    ).all_text_contents()

                    options = [
                        normalize_text(x)
                        for x in options
                        if normalize_text(x)
                    ]

                    candidates.append((i, options))

                except Exception:
                    continue

            month_index = next(
                (
                    i for i, options in candidates
                    if month.lower() in {
                        x.lower() for x in options
                    }
                ),
                None
            )

            if month_index is None:
                print(
                    f"❌ Could not find month '{month}'."
                )
                return False

            print("Selecting month...")

            await selects.nth(month_index).select_option(
                label=month
            )

            await page.wait_for_timeout(2000)

            # Re-discover after month postback.
            frame, selects = await find_date_frame()

            if frame is None:
                print(
                    "❌ Date controls disappeared after month selection."
                )
                return False

            candidates = []

            for i in range(await selects.count()):
                try:
                    options = await selects.nth(i).locator(
                        "option"
                    ).all_text_contents()

                    options = [
                        normalize_text(x)
                        for x in options
                        if normalize_text(x)
                    ]

                    candidates.append((i, options))

                except Exception:
                    continue

            day_index = None

            for i, options in candidates:
                if day in options:
                    numeric_values = []

                    for value in options:
                        if value.isdigit():
                            numeric_values.append(int(value))

                    if numeric_values and max(numeric_values) <= 31:
                        day_index = i
                        break

            if day_index is None:
                print(
                    f"❌ Could not find day '{day}'."
                )
                return False

            print("Selecting day...")

            await selects.nth(day_index).select_option(
                label=day
            )

            await page.wait_for_timeout(3000)

            print(
                "Date selected successfully."
            )

            return True

    # --------------------------------------------------------
    # Fallback: inspect the rendered page for date controls.
    #
    # This is useful if PIB changes from <select> to another
    # control type. We do not silently pretend the date worked.
    # --------------------------------------------------------

    print(
        "\n⚠️ Standard <select> date controls were not found."
    )

    print(
        "Inspecting rendered inputs/buttons for a newer PIB UI..."
    )

    try:
        for frame_number, frame in enumerate(page.frames):

            inputs = frame.locator(
                "input"
            )

            buttons = frame.locator(
                "button"
            )

            print(
                f"\nFrame {frame_number}: "
                f"{await inputs.count()} input(s), "
                f"{await buttons.count()} button(s)"
            )

            for i in range(await inputs.count()):

                element = inputs.nth(i)

                try:
                    input_type = await element.get_attribute("type")
                    name = await element.get_attribute("name")
                    element_id = await element.get_attribute("id")
                    value = await element.get_attribute("value")
                    placeholder = await element.get_attribute("placeholder")

                    print(
                        f"  INPUT {i}: "
                        f"type={input_type}, "
                        f"name={name}, "
                        f"id={element_id}, "
                        f"value={value}, "
                        f"placeholder={placeholder}"
                    )

                except Exception:
                    continue

    except Exception as e:
        print(
            "Debug inspection failed:",
            repr(e)
        )

    # --------------------------------------------------------
    # Save complete page HTML for debugging.
    # --------------------------------------------------------

    try:
        html = await page.content()

        with open(
            "pib_debug_page.html",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(html)

        print(
            "\nSaved rendered PIB HTML to:"
            " pib_debug_page.html"
        )

    except Exception as e:
        print(
            "Could not save debug HTML:",
            repr(e)
        )

    print(
        "\n❌ Could not identify PIB's date controls."
    )

    print(
        "The page structure has changed and needs a new selector."
    )

    return False


def extract_prid(text):

    if not text:
        return None

    match = re.search(
        r"PRID\s*=\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


async def get_release_links(page):

    releases = {}

    print(
        "\nScanning PIB page for release links..."
    )

    for frame_number, frame in enumerate(
        page.frames
    ):

        print(
            f"\nScanning frame {frame_number}:"
        )

        anchors = frame.locator(
            "a"
        )

        count = await anchors.count()

        print(
            f"Found {count} links."
        )

        for i in range(count):

            anchor = anchors.nth(i)

            try:

                text = (
                    await anchor.inner_text()
                ).strip()

            except Exception:

                text = ""

            try:

                href = await anchor.get_attribute(
                    "href"
                )

            except Exception:

                href = ""

            try:

                onclick = await anchor.get_attribute(
                    "onclick"
                )

            except Exception:

                onclick = ""

            combined = " ".join([
                href or "",
                onclick or "",
                text or ""
            ])

            prid = extract_prid(
                combined
            )

            if not prid:
                continue

            url = None

            # ------------------------------------------------
            # HREF
            # ------------------------------------------------

            if href:

                if not href.lower().startswith(
                    "javascript:"
                ):

                    url = urljoin(
                        PIB_BASE,
                        href
                    )

            # ------------------------------------------------
            # ONCLICK
            # ------------------------------------------------

            if not url and onclick:

                match = re.search(
                    r"""['"]([^'"]*PRID[^'"]*)['"]""",
                    onclick,
                    re.IGNORECASE
                )

                if match:

                    url = urljoin(
                        PIB_BASE,
                        match.group(1)
                    )

            # ------------------------------------------------
            # FALLBACK URL
            # ------------------------------------------------

            if not url:

                url = (
                    PIB_BASE
                    + "PressReleseDetailm.aspx"
                    + "?PRID="
                    + prid
                    + "&reg=3&lang=1"
                )

            if "pib.gov.in" not in url.lower():
                continue

            # ------------------------------------------------
            # Store in Python dictionary
            # ------------------------------------------------

            releases[prid] = {
                "release_id": prid,
                "url": url,
                "title": text
            }

    return releases


def crawler_url(url):

    prid = extract_prid(
        url
    )

    if not prid:
        return url

    return (
        PIB_BASE
        + "PressReleasePage.aspx"
        + f"?PRID={prid}&reg=3&lang=1"
    )

