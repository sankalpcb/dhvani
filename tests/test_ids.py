import numpy as np
import pytest
from dhvani.audio import normalize
from dhvani.ids import hypothesis_key, segment_id, variant_slug
from dhvani.config import SAMPLE_RATE


def _tone(seconds=1.0, rate=SAMPLE_RATE, freq=440.0):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_same_audio_yields_same_id():
    pcm = normalize(_tone(), SAMPLE_RATE)
    assert segment_id(pcm) == segment_id(pcm.copy())


def test_id_is_64_char_hex():
    pcm = normalize(_tone(), SAMPLE_RATE)
    sid = segment_id(pcm)
    assert len(sid) == 64
    assert all(c in "0123456789abcdef" for c in sid)


def test_resampling_converges_to_same_id():
    """Same signal captured at 44.1kHz and 16kHz normalizes to near-identical PCM."""
    a = normalize(_tone(rate=44100), 44100)
    b = normalize(_tone(rate=SAMPLE_RATE), SAMPLE_RATE)
    assert len(a) == len(b)
    # Resampling is lossy; assert close, not identical.
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) < 200


def test_different_audio_yields_different_id():
    a = normalize(_tone(freq=440.0), SAMPLE_RATE)
    b = normalize(_tone(freq=880.0), SAMPLE_RATE)
    assert segment_id(a) != segment_id(b)


def test_stereo_is_downmixed_to_mono():
    stereo = np.stack([_tone(), _tone()], axis=1)
    pcm = normalize(stereo, SAMPLE_RATE)
    assert pcm.ndim == 1


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="int16"):
        segment_id(np.zeros(10, dtype=np.float32))


# --- Fix round 2, I2/I3: cache identity is PCM + tier + policy + variant ---

def test_hypothesis_key_separates_tiers():
    assert hypothesis_key("tier0", "lang=hi") != hypothesis_key("tier1", "lang=hi")


def test_hypothesis_key_separates_variants():
    """segment_id hashes PCM alone, so identical audio decoded with a
    different lang or model_id produces different text under the same
    segment_id. The variant must therefore be part of the cache key."""
    assert hypothesis_key("tier0", "lang=hi") != hypothesis_key("tier0", "lang=ml")


def test_hypothesis_key_changes_when_policy_id_is_bumped(monkeypatch):
    """POLICY_ID is documented as the cache-invalidation mechanism (spec
    §3.1, invariant I5). Bumping it must actually change the key."""
    before = hypothesis_key("tier0", "lang=hi")
    monkeypatch.setattr("dhvani.config.POLICY_ID", "p-bumped")
    assert hypothesis_key("tier0", "lang=hi") != before


def test_hypothesis_key_is_stable_for_the_same_inputs():
    assert hypothesis_key("tier0", "lang=hi") == hypothesis_key("tier0", "lang=hi")


def test_variant_slug_is_filesystem_safe():
    """Variant strings contain model ids with '/' in them; the slug is used
    as a single path component and must never introduce a separator."""
    slug = variant_slug("lang=hi;model_id=ai4bharat/indic-conformer-600m")
    assert "/" not in slug and "\\" not in slug
    assert slug and not slug.startswith(".")
    assert all(c.isalnum() or c in "._-" for c in slug)


def test_variant_slug_separates_variants_and_policies(monkeypatch):
    hi = variant_slug("lang=hi")
    assert hi != variant_slug("lang=ml")
    monkeypatch.setattr("dhvani.config.POLICY_ID", "p-bumped")
    assert variant_slug("lang=hi") != hi


def test_variant_slug_does_not_collide_after_sanitizing():
    """Sanitizing unsafe characters to '-' could map two distinct variants
    onto one path; the slug carries a digest of the unsanitized key so it
    cannot."""
    assert variant_slug("lang=hi/x") != variant_slug("lang=hi-x")


# --- source_id: the basename collision ---

def test_two_different_audios_with_the_same_basename_get_different_ids():
    """The bug. dhvani/cli.py used os.path.basename(audio) as the source
    identity, so a/clip.wav and b/clip.wav shared one track history and one
    job namespace inside a single --db: escalating one could surface in the
    other's output, and reconcile() polled across both."""
    from dhvani.ids import source_id

    one = normalize(_tone(freq=200.0), SAMPLE_RATE)
    two = normalize(_tone(freq=880.0), SAMPLE_RATE)

    assert source_id("clip.wav", one) != source_id("clip.wav", two)


def test_the_same_audio_and_name_is_stable_across_directories():
    """The id must depend on the audio and the name, never on where the
    file happens to sit -- re-running the same clip from a different
    working directory has to find its own track, not start a new one."""
    from dhvani.ids import source_id

    pcm = normalize(_tone(freq=200.0), SAMPLE_RATE)
    assert source_id("clip.wav", pcm) == source_id("clip.wav", pcm.copy())


def test_source_id_keeps_the_name_legible():
    """Same tradeoff variant_slug() already makes: source_id lands in the
    segments and jobs tables, and a reviewer reading that DB needs to see
    which file a row belongs to. A bare digest would be unreadable."""
    from dhvani.ids import source_id

    pcm = normalize(_tone(), SAMPLE_RATE)
    assert source_id("interview-01.wav", pcm).startswith("interview-01.wav")


def test_source_id_sanitizes_and_caps_the_name():
    """Names are attacker-adjacent input (they come off the filesystem) and
    end up in a DB column, so they get the same treatment variant_slug
    gives variant strings: sanitized, length-capped, digest appended."""
    from dhvani.ids import source_id

    pcm = normalize(_tone(), SAMPLE_RATE)
    weird = source_id("../../etc/pa sswd?*.wav", pcm)

    assert "/" not in weird and " " not in weird
    assert ".." not in weird
    assert len(weird) < 80


def test_names_that_sanitize_alike_still_differ():
    """Sanitizing is lossy, so two distinct names can collapse to the same
    safe text. The digest covers the name as well as the audio, so they
    still get different ids -- the same reason variant_slug digests the
    UNsanitized string."""
    from dhvani.ids import source_id

    pcm = normalize(_tone(), SAMPLE_RATE)
    assert source_id("a b.wav", pcm) != source_id("a/b.wav", pcm)


def test_source_id_rejects_unnormalized_pcm():
    """Inherited from segment_id: hashing float or stereo audio would make
    the identity depend on decode details rather than on the samples."""
    from dhvani.ids import source_id

    with pytest.raises(ValueError):
        source_id("clip.wav", _tone())
