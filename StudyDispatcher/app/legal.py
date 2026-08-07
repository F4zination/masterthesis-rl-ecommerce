"""HTML pages for the study's legal and participant information."""
from __future__ import annotations

from html import escape

from . import config


_STYLE = """
  :root {
    color-scheme: light;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
      "Segoe UI", sans-serif;
    color: #172033;
    background: #f2f5fb;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: #f2f5fb; line-height: 1.65; }
  main {
    width: min(calc(100% - 32px), 780px);
    margin: 42px auto 24px;
    padding: clamp(26px, 5vw, 48px);
    border: 1px solid #dbe3ef;
    border-radius: 18px;
    background: #fff;
    box-shadow: 0 18px 50px rgba(30, 41, 59, .09);
  }
  h1, h2 { line-height: 1.2; letter-spacing: -.02em; }
  h1 { margin-top: 0; font-size: clamp(2rem, 7vw, 2.7rem); }
  h2 { margin-top: 2rem; font-size: 1.25rem; }
  p, li { color: #475569; }
  a { color: #4338ca; }
  .back { display: inline-block; margin-bottom: 18px; font-weight: 700; }
  .notice {
    padding: 14px 16px;
    border-left: 4px solid #d97706;
    border-radius: 7px;
    background: #fffbeb;
    color: #78350f;
  }
  .missing { color: #b91c1c; font-weight: 800; }
  footer {
    width: min(calc(100% - 32px), 780px);
    margin: 0 auto 38px;
    color: #64748b;
    font-size: .86rem;
    text-align: center;
  }
  footer a { margin: 0 8px; }
"""


def _field(value: str, description: str) -> str:
    if value:
        return escape(value).replace("\n", "<br>")
    return f'<span class="missing">[Configure {escape(description)} before deployment]</span>'


def _footer() -> str:
    researcher = escape(config.STUDY_RESEARCHER_NAME or "Study project")
    return f"""
<footer>
  <div>
    <a href="/imprint">Legal notice</a>
    <a href="/privacy">Privacy</a>
    <a href="/participant-information">Participant information</a>
  </div>
  <p>&copy; 2026 {researcher} &middot; Master thesis study</p>
</footer>"""


def is_shop_host(request) -> bool:
    """Return whether this legal page is being served on a demo-shop hostname.

    The gateway proxies ``/imprint``, ``/privacy`` and ``/participant-information``
    on both condition domains, so the same page is reachable from inside a running
    shopping session. On those hosts the "back" link must return the participant to
    the shop — sending a mid-session participant to the entry form let them
    re-register under a new nickname, which minted a second assignment and skewed
    the balanced allocation.
    """
    host = (request.headers.get("host") or "").split(":", 1)[0].lower()
    if not host:
        return False
    return host in config.shop_hostnames()


def _page(title: str, content: str, *, on_shop_host: bool = False) -> str:
    back = (
        '<a class="back" href="/">&larr; Back to the shop</a>'
        if on_shop_host
        else '<a class="back" href="/">&larr; Back to study entry</a>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} &ndash; Shopping Study</title>
  <style>{_STYLE}</style>
</head>
<body>
  <main>
    {back}
    {content}
  </main>
  {_footer()}
</body>
</html>"""


def message_page(title: str, message: str, *, on_shop_host: bool = False) -> str:
    """Render a small status/error page with the permanent legal footer."""
    return _page(
        title,
        f"<h1>{escape(title)}</h1><p>{escape(message)}</p>",
        on_shop_host=on_shop_host,
    )


def imprint_page(*, on_shop_host: bool = False) -> str:
    """Render the provider information customarily required by § 5 DDG."""
    controller = _field(config.LEGAL_CONTROLLER_NAME, "LEGAL_CONTROLLER_NAME")
    address = _field(config.LEGAL_CONTROLLER_ADDRESS, "LEGAL_CONTROLLER_ADDRESS")
    email = _field(config.LEGAL_CONTROLLER_EMAIL, "LEGAL_CONTROLLER_EMAIL")
    phone = escape(config.LEGAL_CONTROLLER_PHONE) if config.LEGAL_CONTROLLER_PHONE else "Not provided"
    responsible = _field(
        config.LEGAL_CONTENT_RESPONSIBLE or config.LEGAL_CONTROLLER_NAME,
        "LEGAL_CONTENT_RESPONSIBLE",
    )
    institution = escape(config.STUDY_INSTITUTION or "Not specified")
    additional = ""
    if config.LEGAL_ADDITIONAL_IMPRINT_DETAILS:
        additional = (
            "<h2>Additional provider details</h2><p>"
            + escape(config.LEGAL_ADDITIONAL_IMPRINT_DETAILS).replace("\n", "<br>")
            + "</p>"
        )
    warning = ""
    if not config.legal_information_complete():
        warning = """
<p class="notice"><strong>Deployment notice:</strong> The highlighted fields must be
completed with the actual legally responsible person or organisation before recruiting
participants. The university must only be named as controller if it has confirmed that role.</p>"""
    return _page(
        "Legal notice (Impressum)",
        f"""
<h1>Legal notice (Impressum)</h1>
{warning}
<h2>Provider information pursuant to § 5 DDG</h2>
<p>{controller}<br>{address}</p>
<p>Email: {email}<br>Telephone: {phone}</p>
<h2>Study project</h2>
<p>This website is operated for a master thesis study by
{escape(config.STUDY_RESEARCHER_NAME or "the named researcher")} at {institution}.</p>
<h2>Responsible for content</h2>
<p>{responsible}<br>{address}</p>
{additional}
<h2>No online shop or consumer contract</h2>
<p>The displayed shop is a research simulation. No real purchase, payment, shipment,
guarantee, or consumer contract is offered or concluded.</p>
""",
        on_shop_host=on_shop_host,
    )


def privacy_page(*, on_shop_host: bool = False) -> str:
    """Render the Article 13 GDPR privacy information for the study."""
    controller = _field(config.LEGAL_CONTROLLER_NAME, "LEGAL_CONTROLLER_NAME")
    address = _field(config.LEGAL_CONTROLLER_ADDRESS, "LEGAL_CONTROLLER_ADDRESS")
    email = _field(config.LEGAL_CONTROLLER_EMAIL, "LEGAL_CONTROLLER_EMAIL")
    hosting = _field(config.LEGAL_HOSTING_PROVIDER, "LEGAL_HOSTING_PROVIDER")
    retention = _field(config.STUDY_RETENTION_PERIOD, "STUDY_RETENTION_PERIOD")
    log_retention = _field(
        config.SERVER_LOG_RETENTION_PERIOD, "SERVER_LOG_RETENTION_PERIOD"
    )
    third_country = _field(
        config.LEGAL_THIRD_COUNTRY_INFORMATION,
        "LEGAL_THIRD_COUNTRY_INFORMATION",
    )
    if config.LEGAL_DATA_PROTECTION_CONTACT:
        dpo = escape(config.LEGAL_DATA_PROTECTION_CONTACT).replace("\n", "<br>")
    else:
        dpo = (
            "No contact has been configured. If the university is the controller, "
            "its data-protection officer must be listed here."
        )
    return _page(
        "Privacy notice",
        f"""
<h1>Privacy notice</h1>
<p>Information pursuant to Articles 12 and 13 of the General Data Protection Regulation
(GDPR), version {escape(config.CONSENT_VERSION)}.</p>

<h2>1. Controller and contact</h2>
<p>{controller}<br>{address}<br>Email: {email}</p>
<p><strong>Data-protection contact:</strong><br>{dpo}</p>

<h2>2. Purpose and legal basis</h2>
<p>The study compares two adaptive e-commerce systems and examines how algorithmically
selected interface elements affect navigation and shopping decisions. Study data and
browser storage are processed on the basis of your voluntary consent (Article 6(1)(a)
GDPR and § 25(1) TDDDG). Security and access logs required to operate the service are
processed to provide and protect the website (Article 6(1)(f) GDPR, where applicable).</p>

<h2>3. Data processed</h2>
<ul>
  <li>the nickname you choose, the generated participant and session identifiers,
      assignment to a study condition and scenario, consent version and time;</li>
  <li>page views, clicks, scrolling, dwell time, cart and simulated-order actions,
      attention-check answers, completion code, and timestamps;</li>
  <li>algorithm decisions, displayed interface elements, interaction outcomes,
      propensities, and contextual attributes used by the research systems;</li>
  <li>browser and device information such as user agent, referrer, screen dimensions,
      and technical access data such as IP address in server logs.</li>
</ul>
<p>Do not use your real name as a nickname and do not enter real names, addresses, email
addresses, or payment information in the simulated shop. Checkout field values are not
evaluated or saved by the study application.</p>

<h2>4. Pseudonymization and publication</h2>
<p>During collection, records are linked by participant and session identifiers and are
therefore <strong>pseudonymized, not anonymous</strong>. Direct identifiers are not
required. Publications and thesis results use only aggregated or anonymized data from
which participants are not reasonably identifiable.</p>

<h2>5. Cookies and browser storage</h2>
<ul>
  <li><code>wid</code> and <code>study_persona</code>: study attribution, up to 1 hour;</li>
  <li><code>session_id</code>: connects actions within the simulated shop, up to 30 days;</li>
  <li>local storage entries for study start time and page count: remain until browser
      storage is cleared;</li>
  <li>session storage for page depth and serving controls: until the browser tab is closed.</li>
</ul>

<h2>6. Recipients, hosting, and third-party resources</h2>
<p>Access is limited to the researcher and authorised academic supervisors as required
for the thesis. The website is hosted by: {hosting}. The shop currently loads Bootstrap
and Bootstrap Icons from the jsDelivr content-delivery network; the provider therefore
receives technical request data and may process it through infrastructure outside the
EU/EEA. Information about any third-country transfer and safeguards: {third_country}.
No study dataset is sold or used for advertising.</p>

<h2>7. Automated personalization</h2>
<p>Algorithms select recommendations, offers, help elements, or other shop widgets using
session behaviour. This is part of the experiment and has no legal or similarly significant
effect. No real prices, purchases, credit decisions, or contractual decisions are made.</p>

<h2>8. Retention and anonymization</h2>
<p><strong>Study records:</strong> {retention}</p>
<p><strong>Technical server logs:</strong> {log_retention}</p>

<h2>9. Voluntary participation and withdrawal</h2>
<p>Providing study data is not legally or contractually required. Without consent you
cannot participate, but refusing or stopping causes no disadvantage outside any separate
compensation rules of the recruitment platform. You may withdraw consent at any time by
contacting {email} and providing your participant ID. Withdrawal does not affect processing
that was lawful before withdrawal. Once data have been irreversibly anonymized, individual
records can no longer be located or deleted.</p>

<h2>10. Your rights</h2>
<p>Subject to the legal conditions, you have rights of access, rectification, erasure,
restriction, data portability, objection to processing based on legitimate interests,
and withdrawal of consent. You may also complain to a data-protection supervisory
authority, including the
<a href="https://www.baden-wuerttemberg.datenschutz.de/" rel="external">State Commissioner
for Data Protection and Freedom of Information Baden-Württemberg</a>.</p>
""",
        on_shop_host=on_shop_host,
    )


def participant_information_page(*, on_shop_host: bool = False) -> str:
    """Render concise informed-consent information for participants."""
    email = _field(config.LEGAL_CONTROLLER_EMAIL, "LEGAL_CONTROLLER_EMAIL")
    researcher = escape(config.STUDY_RESEARCHER_NAME or "the researcher")
    institution = escape(config.STUDY_INSTITUTION or "the named institution")
    return _page(
        "Participant information",
        f"""
<h1>Participant information</h1>
<h2>What is this study about?</h2>
<p>{researcher} is conducting this master thesis study at {institution}. It compares two
systems that adapt a simulated online shop and investigates navigation and shopping behaviour.</p>

<h2>What will I do?</h2>
<p>You will receive a scenario, browse the simulated shop, and may place a fictional order.
Your interactions are recorded automatically. At the end, you complete a short attention
check and receive a completion code. No real purchase or payment takes place.</p>

<h2>What data are collected?</h2>
<p>A pseudonymous participant/session ID, study assignment, clicks, viewed pages, scrolling,
dwell time, cart and simulated checkout activity, attention answers, device/browser context,
and the adaptive system's decisions. Please use only a fictional nickname and fictional
checkout details. See the <a href="/privacy">full privacy notice</a> for exact storage,
recipient, retention, and rights information.</p>

<h2>Voluntary consent</h2>
<p>Participation is voluntary. You can leave by closing the website and can withdraw your
consent by emailing {email} with your participant ID until the data are irreversibly
anonymized. Refusal or withdrawal causes no disadvantage outside any separate recruitment-
platform rules. Only anonymized or aggregated results will be published.</p>

<h2>Risks and contact</h2>
<p>No risks beyond ordinary website use are expected. The simulated shop may display
personalized offers or recommendations. For study or data-protection questions, contact
{email} before participating.</p>
""",
        on_shop_host=on_shop_host,
    )
