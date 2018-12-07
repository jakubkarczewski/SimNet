import argparse

from content_based import ContentBasedRecommender

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to features .csv file.")
    parser.add_argument("--keywords_path", type=str, required=True,
                        help="Path to keywords .csv file.")
    parser.add_argument("--title", type=str, required=True,
                        help="Title (in double quotes) for which recommendation takes place.")
    parser.add_argument("--top_n", type=int, default=7,
                        help="Show top N recommendations.")
    parser.add_argument("--enable_cache", type=bool, default=True,
                        help="Cache flag.")
    args = parser.parse_args()

    recommender = ContentBasedRecommender(args.features_path, args.keywords_path, args.enable_cache)
    results = recommender.get_recommendation(args.title)
    print('Recommended movies for {}:'. format(args.title))
    for i, result in enumerate(results):
        if args.top_n == i:
            break
        print('nr. {} -> {}'.format(str(i), result))
