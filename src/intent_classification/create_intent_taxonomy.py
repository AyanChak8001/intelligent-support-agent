INTENT_TAXONOMY = {
    "APP_TECHNICAL_COMPATIBILITY": {
        "description": "Problems with the Spotify app, device compatibility, operating systems, crashes, updates, or technical behavior.",
        "examples": [
            "Spotify does not work on my iPhone",
            "The app keeps crashing",
            "Please update Spotify for my device"
        ]
    },

    "SUBSCRIPTION_PLAN": {
        "description": "Questions or problems related to Premium plans, Student plans, Family plans, eligibility, or subscriptions.",
        "examples": [
            "My student discount is not working",
            "Family plan redeem code does not work",
            "Am I eligible for this offer?"
        ]
    },

    "BILLING_PAYMENT_REFUND": {
        "description": "Problems with charges, payments, billing, refunds, or incorrect pricing.",
        "examples": [
            "I was charged twice",
            "I need a refund",
            "Why was I charged the full price?"
        ]
    },

    "PLAYLIST_LIBRARY_DISCOVERY": {
        "description": "Problems or questions related to playlists, music library, queue, search, recommendations, or discovering music.",
        "examples": [
            "How do I delete songs from my playlist?",
            "My library is missing songs",
            "How does the queue work?"
        ]
    },

    "DOWNLOADS_OFFLINE_STORAGE": {
        "description": "Problems related to downloaded music, offline tracks, storage usage, or SD cards.",
        "examples": [
            "My offline tracks disappeared",
            "Spotify is using too much storage",
            "Downloaded songs will not delete"
        ]
    },

    "ACCOUNT_ACCESS_SECURITY": {
        "description": "Problems related to login, account access, passwords, account security, or unauthorized account usage.",
        "examples": [
            "Someone is using my account",
            "I cannot log in",
            "My account was hacked"
        ]
    },

    "PLAYBACK_AUDIO": {
        "description": "Problems with playing music, audio interruptions, sound, songs stopping, or playback behavior.",
        "examples": [
            "Music keeps stopping",
            "Songs will not play",
            "My music cuts off unexpectedly"
        ]
    },

    "COMMUNITY_INTEGRATIONS": {
        "description": "Problems related to internet connectivity, third-party services, devices, or external integrations.",
        "examples": [
            "Spotify will not connect",
            "Spotify does not work with my device",
            "Integration is not working"
        ]
    },

    "CONTENT_AVAILABILITY_METADATA": {
        "description": "Questions about missing songs, albums, artists, incorrect music information, or content availability.",
        "examples": [
            "Why is this album unavailable?",
            "This song is missing",
            "Why is this artist information incorrect?"
        ]
    },

    "FEATURE_REQUEST_PRODUCT_FEEDBACK": {
        "description": "Requests for new features, product improvements, suggestions, or general feedback.",
        "examples": [
            "Please add this feature",
            "Why did you remove this feature?",
            "It would be nice to have this option"
        ]
    },

    "OTHER_UNCLEAR": {
        "description": "Messages that are too vague, conversational, incomplete, or impossible to classify without additional context.",
        "examples": [
            "Thanks",
            "Same problem",
            "Can you help me?"
        ]
    }
}


def main():
    print("=" * 70)
    print("FINAL INTENT TAXONOMY")
    print("=" * 70)

    for intent, details in INTENT_TAXONOMY.items():
        print(f"\n{intent}")
        print(f"Description: {details['description']}")

        print("Examples:")
        for example in details["examples"]:
            print(f"  - {example}")


if __name__ == "__main__":
    main()
