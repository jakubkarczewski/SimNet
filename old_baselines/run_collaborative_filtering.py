import argparse

from collaborative_filtering import CFLSHRecommender, CFRecommender

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_path", type=str, required=True,
                        help="Path to ratings .csv file.")
    parser.add_argument("--titles_path", type=str, required=True,
                        help="Path to titles .csv file.")
    parser.add_argument("--user_ratings_path", type=str, required=True,
                        help="Path to user ratings .csv file made from template.")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Show top N recommendations.")
    parser.add_argument("--recommender_type", type=str, required=True,
                        help="lsh or standard")
    parser.add_argument("--gen_dataset", type=bool, default=False,
                        help="True if generating dataset for NN training.")
    parser.add_argument("--num_permutations", type=int, default=128,
                        help="Number of permutations.")
    parser.add_argument("--show_mem_usage", type=bool, default=True,
                        help="Number of permutations.")
    parser.add_argument("--lsh_per_movie", type=bool, default=False,
                        help="Return N recommendation per each movie if True.")

    args = parser.parse_args()

    if args.recommender_type == 'standard':
        recommender = CFRecommender(args.ratings_path, args.titles_path, args.user_ratings_path)
    elif args.recommender_type == 'lsh':
        recommender = CFLSHRecommender(args.ratings_path, args.titles_path, args.user_ratings_path,
                                       args.num_permutations)
    else:
        raise Exception('Recommender type must be either standard or lsh.')

    if not args.gen_dataset:
        results = recommender.get_recommendation(top_n=args.top_n)
        if not args.lsh_per_movie:
            results = recommender.reduce_to_n(results, args.top_n)
        print('Recommendations:')

        if args.recommender_type == 'standard' or not args.lsh_per_movie:
            for i, result in enumerate(results):
                print('nr. {} -> {}'.format(str(i + 1), result))
        else:
            for title in results.keys():
                print('Based on {} :'.format(title), results[title])
    else:
        recommender.generate_dataset(show_mem_usage=args.show_mem_usage)
