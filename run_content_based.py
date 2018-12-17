import argparse

from content_based import ContentBasedRecommender, ContentBasedLSHRecommender

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
    parser.add_argument("--recommender_type", type=str, required=True,
                        help="lsh or cosine")
    parser.add_argument("--gen_dataset", type=bool, default=False,
                        help="True if generating dataset for NN training.")
    parser.add_argument("--num_permutations", type=int, default=128,
                        help="Number of permutations.")
    args = parser.parse_args()

    if args.recommender_type == 'cosine':
        recommender = ContentBasedRecommender(args.features_path, args.keywords_path, args.enable_cache)
    elif args.recommender_type == 'lsh':
        recommender = ContentBasedLSHRecommender(args.features_path, args.keywords_path, args.num_permutations,
                                                 args.enable_cache)
    else:
        raise Exception('Recommender type must be either cosine or lsh.')
    if not args.gen_dataset:
        results = recommender.get_recommendation(args.title, num_results=args.top_n)
        print('Recommended movies for {}:'. format(args.title))
        for i, result in enumerate(results):
            print('nr. {} -> {}'.format(str(i), result))
    else:
        recommender.get_all_similarities()
