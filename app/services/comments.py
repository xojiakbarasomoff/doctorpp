"""Comments under the doctor's posts: answered in Direct, acknowledged in public.

A comment is read by everyone who opens the post, so the answer -- which is
about somebody's body -- goes to their inbox instead, and the public reply
says only that it is waiting there. Instagram allows exactly one such private
reply per comment, which is why it carries the answer and the public line
carries none of it.
"""

from app.services.language import reply_script

PUBLIC_REPLIES = {
    "uz-latn": "Assalomu alaykum! Savolingizga Direct orqali javob berdik 🙌",
    "uz-cyrl": "Ассалому алайкум! Саволингизга Директ орқали жавоб бердик 🙌",
    "ru": "Здравствуйте! Мы ответили вам в Директ 🙌",
}


def public_reply(comment_text: str) -> str:
    """The line left under the comment, in the script it was written in."""
    return PUBLIC_REPLIES[reply_script(comment_text)]
