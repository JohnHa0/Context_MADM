
class PLTS:
    def __init__(self, terms, max_scale=7, symmetric_scale=True, weight=1):
        self.terms = terms
        self.max_scale = (
            max_scale  # The maximum scale index used for flipping the scale
        )
        self.symmetric_scale = (
            symmetric_scale  # True if the scale has both positive and negative values
        )
        # self.weight = weight
        self.normalize()

    def normalize(self):
        # total_prob = sum(self.terms.values())
        # self.terms = {scale: prob / total_prob for scale, prob in self.terms.items()}
        # """Normalize the PLTS only if it has more than one element."""
        if len(self.terms) > 1:
            total_prob = sum(self.terms.values())
            if total_prob > 0:
                self.terms = {
                    scale: prob / total_prob for scale, prob in self.terms.items()
                }
            # else:
            # If the total probability is zero, assign a neutral term
            # self.terms = {"s0": 1.0}

    def apply_weight(self, weight):
        """Apply a weight to all scale indices in the PLTS."""
        weighted_terms = {}
        for scale, prob in self.terms.items():
            scale_index = self.extract_scale_index(scale)
            weighted_index = scale_index * weight
            weighted_scale = (
                f"s{weighted_index}"
                if weighted_index >= 0
                else f"s-{abs(weighted_index)}"
            )
            weighted_terms[weighted_scale] = prob
        return PLTS(weighted_terms, self.max_scale, self.symmetric_scale)

    def flip_scale(self, scale):
        # Flip the scale index according to the rules for positive and negative indices
        index = self.extract_scale_index(scale)
        if self.symmetric_scale:
            # For symmetric scales, flip the scale by negating the index
            return "s" + str(0 - index)
        else:
            # For all positive scales, flip the scale within the range
            return "s" + str(self.max_scale - index - 1)

    def extract_scale_index(self, scale):
        """从标度字符串中提取下标，考虑负数情况"""
        if scale.startswith("s-"):
            return float(scale[1:])
        else:
            return float(scale[1:])

    def multiply_with(self, other, weight_factor=1.5):
        """乘法运算，并应用权重因子"""
        combined_terms = self.terms.copy()
        for scale, prob in other.terms.items():
            scale_index = self.extract_scale_index(scale)
            if scale in combined_terms:
                combined_terms[scale] *= prob * weight_factor
            else:
                combined_terms[scale] = prob
        result = PLTS(combined_terms)
        # result.normalize()
        return result

    def add_with(self, other, weight_factor=1):
        """加法运算，并应用权重因子"""
        combined_terms = self.terms.copy()
        for scale, prob in other.terms.items():
            scale_index = self.extract_scale_index(scale)
            if scale in combined_terms:
                # a+b
                # combined_terms[scale] += prob * weight_factor
                # 使用a+b-a*b形式计算概率
                combined_terms[scale] = (
                    combined_terms[scale] + prob - combined_terms[scale] * prob
                )
            else:
                combined_terms[scale] = prob
        result = PLTS(combined_terms)
        # result.normalize()
        return result

    def subtract_with(self, other):
        """Subtract another PLTS from this one."""
        result_terms = self.terms.copy()

        # Subtract matching scales using the a - a*b rule
        for scale, prob in other.terms.items():
            if scale in result_terms:
                result_terms[scale] -= result_terms[scale] * prob
            else:
                # Flip the scale index for non-matching scales
                flipped_scale = self.flip_scale(scale)
                # If the flipped scale is present, subtract from it
                if flipped_scale in result_terms:
                    result_terms[flipped_scale] -= result_terms[flipped_scale] * prob
                else:
                    # If not, add the flipped scale with a negative probability
                    result_terms[flipped_scale] = prob

        # Normalize the result to remove negative probabilities and ensure the sum is 1
        result_plts = PLTS(result_terms, self.max_scale, self.symmetric_scale)
        result_plts.normalize()
        return result_plts

    def score(self):
        """计算PLTS的加权平均"""
        weighted_sum = sum(
            self.extract_scale_index(scale) * prob for scale, prob in self.terms.items()
        )
        return weighted_sum

    @staticmethod
    def operate_all(plts_list, operation, weight_factor=1.5):
        """对多个PLTS进行操作（乘法或加法）"""
        if not plts_list:
            return PLTS({})

        result = plts_list[0]
        for plts in plts_list[1:]:
            result = operation(result, plts, weight_factor)

        result.normalize()
        return result

    @staticmethod
    def traverse_addition(plts_list):
        """对PLTS的所有元素进行遍历加法，包括语言标度的下标相加"""
        combined_terms = {}
        for i, plts1 in enumerate(plts_list):
            for scale1, prob1 in plts1.terms.items():
                for j, plts2 in enumerate(plts_list):
                    if i != j:  # 确保不与自身组合
                        for scale2, prob2 in plts2.terms.items():
                            scale_index1 = plts1.extract_scale_index(scale1)
                            scale_index2 = plts2.extract_scale_index(scale2)
                            new_scale_index = scale_index1 + scale_index2
                            new_scale = "s" + str(new_scale_index)
                            # combined_terms[new_scale] = (
                            #     combined_terms.get(new_scale, 0) + prob1 + prob2
                            # )
                            # a+b-a*b
                            combined_terms[new_scale] = (
                                combined_terms.get(new_scale, 0)
                                + prob1
                                + prob2
                                - combined_terms.get(new_scale, 0) * (prob1 + prob2)
                            )

        result = PLTS(combined_terms)
        result.normalize()
        return result




class PLTSEvaluationMatrix:
    def __init__(self, plts_matrix, weights, max_scale=7, symmetric_scale=True):
        self.plts_matrix = (
            plts_matrix  # Matrix of PLTS objects for each scheme and criterion
        )
        self.weights = weights  # Weights for each criterion
        self.max_scale = max_scale  # Maximum scale index
        self.symmetric_scale = (
            symmetric_scale  # Indicates if scales include negative values
        )

    def calculate_SM_for_scheme(self, scheme_plts):
        """Calculate the evaluation matrix for a single scheme according to the formula"""
        num_criteria = len(self.weights)

        scheme_matrix = np.empty((num_criteria, num_criteria), dtype=object)

        score_matrix = np.zeros((num_criteria, num_criteria))

        for j in range(num_criteria):
            for k in range(num_criteria):
                # Calculate the score for each PLTS
                xij_score = scheme_plts[j].score()
                xik_score = scheme_plts[k].score()

                # Calculate the terms based on the given formula
                if j != k:
                    diff = (self.weights[j] * xij_score) - (self.weights[k] * xik_score)
                    # term = f"s{diff}" if diff >= 0 else f"s-{abs(diff)}"
                    if diff >= 0:
                        a1 = scheme_plts[j].apply_weight(self.weights[j])
                        a2 = scheme_plts[k].apply_weight(self.weights[k])
                        scheme_matrix[j][k] = a1.subtract_with(a2)
                    else:
                        # pass
                        scheme_matrix[j][k] = PLTS({"s0": 0})
                else:
                    if xij_score >= 0:
                        # pass
                        scheme_matrix[j][k] = scheme_plts[j]
                    else:
                        # pass
                        scheme_matrix[j][k] = PLTS({"s0": 0})

                scheme_matrix[j][k].normalize()  # Normalize the PLTS in the matrix

        for i in range(num_criteria):
            for j in range(num_criteria):
                score_matrix[i, j] = scheme_matrix[i, j].score()
                # if score_matrix[i, j]:
                #     score_matrix[i, j] = scheme_matrix[i, j].score()
                # else:
                #     score_matrix[i, j] = 0

            return scheme_matrix, score_matrix

    def calculate_IM_for_scheme(self, scheme_plts):
        """Calculate the evaluation matrix for a single scheme according to the formula"""
        num_criteria = len(self.weights)
        scheme_matrix = np.empty((num_criteria, num_criteria), dtype=object)
        score_matrix = np.zeros((num_criteria, num_criteria))

        for j in range(num_criteria):
            for k in range(num_criteria):
                # Calculate the score for each PLTS
                xij_score = scheme_plts[j].score()
                xik_score = scheme_plts[k].score()

                # Calculate the terms based on the given formula
                if j != k:
                    diff = (self.weights[j] * xij_score) - (self.weights[k] * xik_score)
                    # term = f"s{diff}" if diff >= 0 else f"s-{abs(diff)}"
                    if diff <= 0:
                        a1 = scheme_plts[j].apply_weight(self.weights[j])
                        a2 = scheme_plts[k].apply_weight(self.weights[k])
                        scheme_matrix[j][k] = a1.subtract_with(a2)
                    else:
                        scheme_matrix[j][k] = PLTS({"s0": 0})
                        # pass
                else:
                    if xij_score >= 0:
                        # scheme_matrix[j][k] = PLTS({"sSelf": 1})
                        scheme_matrix[j][k] = scheme_plts[j]
                        # pass
                    else:
                        # pass
                        scheme_matrix[j][k] = PLTS({"s0": 0})
                        # scheme_matrix[j][k] = PLTS({"sNon": 1})

                scheme_matrix[j][k].normalize()  # Normalize the PLTS in the matrix

        for i in range(num_criteria):
            for j in range(num_criteria):
                score_matrix[i, j] = scheme_matrix[i, j].score()
                # if score_matrix[i, j]:
                #     score_matrix[i, j] = scheme_matrix[i, j].score()
                # else:
                #     score_matrix[i, j] = 0

        return scheme_matrix, score_matrix

    def calculate_matrices_SM(self):
        """Calculate the evaluation matrices for all schemes"""
        num_schemes = len(self.plts_matrix)
        num_criteria = len(self.weights)
        # all_matrices = np.empty([num_schemes, num_criteria, num_criteria])
        all_matrices = []

        for i in range(num_schemes):
            _, scheme_matrix = self.calculate_SM_for_scheme(self.plts_matrix[i])
            all_matrices.append(scheme_matrix)
            # all_matrices[i, :, :] = scheme_matrix

        return all_matrices

    def calculate_matrices_IM(self):
        """Calculate the evaluation matrices for all schemes"""
        num_schemes = len(self.plts_matrix)
        all_matrices = []

        for i in range(num_schemes):
            _, scheme_matrix = self.calculate_IM_for_scheme(self.plts_matrix[i])
            all_matrices.append(scheme_matrix)

        return all_matrices

    def sum_row_plts(self, row_index):
        """Calculate the sum of all PLTS objects in a given row."""
        row_sum = PLTS(
            {}, self.max_scale, self.symmetric_scale
        )  # Start with an empty PLTS
        for plts in self.plts_matrix[row_index]:
            row_sum = row_sum.add_plts(plts)  # Add each PLTS in the row
        row_sum.normalize()  # Normalize the sum
        return row_sum

    def sum_row_plts_store(self):
        """Calculate the sum of all PLTS objects in each row and store in a matrix."""
        num_rows = len(self.plts_matrix)
        row_sums_matrix = np.empty((num_rows,), dtype=object)

        for i in range(num_rows):
            row_sum = PLTS({}, self.max_scale, self.symmetric_scale)
            for plts in self.plts_matrix[i]:
                row_sum = row_sum.add_plts(plts)
            row_sum.normalize()
            row_sums_matrix[i] = row_sum

        return row_sums_matrix

    def sum_all_plts(self):
        """Calculate the sum of all PLTS objects in the matrix."""
        total_sum = PLTS(
            {}, self.max_scale, self.symmetric_scale
        )  # Start with an empty PLTS
        for row in self.plts_matrix:
            for plts in row:
                total_sum = total_sum.add_plts(plts)  # Add each PLTS in the matrix
        total_sum.normalize()  # Normalize the sum
        return total_sum



class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def sum_rows_in_matrices(self):
        num_matrices = len(self.matrix)
        num_rows = len(self.matrix[0])
        row_sums = []
        for matrix in self.matrix:
            for row in matrix:
                row_sum = PLTS({}, self.max_scale, self.symmetric_scale)
                for plts in row:
                    row_sum = row_sum.add_plts(plts)
                row_sum.normalize()
                row_sums.append(row_sum)
        return row_sums

    def sum_all_plts_in_matrices(self):
        total_sum = PLTS({}, self.max_scale, self.symmetric_scale)
        for matrix in self.matrix:
            for row in matrix:
                for plts in row:
                    total_sum = total_sum.add_plts(plts)
        total_sum.normalize()
        return total_sum
