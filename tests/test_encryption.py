import pytest

from app.core.encryption import DecryptionError, decrypt, encrypt

# Two distinct, real (test-only) Fernet keys — all tests pass an explicit
# key= rather than depending on ENCRYPTION_KEY/get_settings(), so these stay
# deterministic regardless of what's in the real .env.
_KEY_A = "Hq3_REB-V0twf7iBgCPCSUZQiG44egxyiZg9kOKRxUg="
_KEY_B = "r-13SHrRMryXJLKCg9qizGYe1EnDGBhHMy5YT3Bcz9M="


def test_encrypt_then_decrypt_round_trips() -> None:
    plaintext = "EAAG...a-real-looking-instagram-access-token"
    ciphertext = encrypt(plaintext, key=_KEY_A)
    assert decrypt(ciphertext, key=_KEY_A) == plaintext


def test_ciphertext_is_not_the_plaintext() -> None:
    ciphertext = encrypt("secret-token", key=_KEY_A)
    assert ciphertext != "secret-token"


def test_decrypt_with_wrong_key_raises_decryption_error() -> None:
    ciphertext = encrypt("secret-token", key=_KEY_A)
    with pytest.raises(DecryptionError):
        decrypt(ciphertext, key=_KEY_B)


def test_decrypt_of_never_encrypted_string_raises_decryption_error() -> None:
    # Guards the "credentials went in as plaintext by mistake" failure mode:
    # this must raise, not silently hand back the garbage input as if it
    # were the real token.
    with pytest.raises(DecryptionError):
        decrypt("this-was-never-encrypted", key=_KEY_A)


def test_decrypt_of_empty_string_raises_decryption_error() -> None:
    with pytest.raises(DecryptionError):
        decrypt("", key=_KEY_A)
